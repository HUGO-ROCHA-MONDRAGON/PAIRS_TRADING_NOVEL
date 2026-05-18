"""
bloomberg/blp.py
================
Les trois fonctions Bloomberg : bdh, bdp, bds.

Améliorations par rapport au blp_bds.py original, inspirées de xbbg :

1. Overrides Bloomberg via **kwargs
   Syntaxe directe :  bds('SPX Index', 'INDX_MWEIGHT_HIST', END_DATE_OVERRIDE='20240101')
   Pas besoin de dictionnaire séparé ni de paramètres nommés en dur.

2. Batching automatique
   bdh : CHUNK_BDH tickers par HistoricalDataRequest (limite Bloomberg ~25)
   bdp : CHUNK_BDP tickers par ReferenceDataRequest  (limite Bloomberg ~100)
   Même un appel sur 500 tickers S&P 500 est découpé et recomposé proprement.

3. Boucle d'événements unifiée (_collect)
   Plus aucun copier-coller du while True / nextEvent() entre les trois fonctions.

4. Sorties cohérentes
   bdh → MultiIndex colonnes (ticker, field), DatetimeIndex en index
   bdp → index = security, colonnes = fields (noms en lowercase)
   bds → DataFrame plat, noms de colonnes Bloomberg préservés

5. Inputs flexibles
   tickers/flds : str ou list, toujours normalisé
   dates        : str 'YYYY-MM-DD', datetime, pd.Timestamp → converti en 'YYYYMMDD'
"""
from __future__ import annotations

from typing import List, Optional, Union

import blpapi  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from ._session import get_session

# Bloomberg limite selon le type de requête :
#   HistoricalDataRequest  : ~25 securities, ~25 fields
#   ReferenceDataRequest   : ~100 securities, ~100 fields
# On utilise des marges confortables.
CHUNK_BDH: int = 25
CHUNK_BDP: int = 100

DateLike = Union[str, pd.Timestamp]


# ──────────────────────────────────────── Helpers internes ────────────────────


def _to_blp_date(d: DateLike) -> str:
    """Convertit toute date en 'YYYYMMDD' (format attendu par Bloomberg)."""
    return pd.to_datetime(d).strftime("%Y%m%d")


def _as_list(x) -> list:
    """Str ou itérable → toujours list."""
    return [x] if isinstance(x, str) else list(x)


def _chunked(lst: list, n: int):
    """Générateur de sous-listes de taille n."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _collect(session) -> list:
    """
    Collecte tous les messages Bloomberg jusqu'à l'événement RESPONSE final.

    Inspiré de xbbg : une seule boucle partagée entre bdh, bdp et bds.
    Gère les PARTIAL_RESPONSE (requêtes qui arrivent en plusieurs morceaux).
    """
    msgs: list = []
    while True:
        event = session.next_event()
        etype = event.eventType()
        if etype in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
            msgs.extend(list(event))
        if etype == blpapi.Event.RESPONSE:
            break
    return msgs


def _set_overrides(req, overrides: dict) -> None:
    """
    Applique des overrides Bloomberg sur une requête.

    Conversion directe de **kwargs en éléments 'overrides' Bloomberg.
    C'est la syntaxe xbbg : bdp('AAPL US Equity', 'Eqy_Weighted_Avg_Px', VWAP_Dt='20240115')
    """
    if not overrides:
        return
    ovrd_elem = req.getElement("overrides")
    for key, val in overrides.items():
        o = ovrd_elem.appendElement()
        o.setElement("fieldId", key)
        o.setElement("value", str(val))


# ──────────────────────────────────────────────────────────────── BDH ─────────


def bdh(
    tickers: Union[str, List[str]],
    flds: Union[str, List[str]],
    start_date: DateLike,
    end_date: DateLike,
    Per: str = "DAILY",
    Fill: str = "PREVIOUS_VALUE",
    Days: str = "NON_TRADING_WEEKDAYS",
    currency: Optional[str] = None,
    adjust: Optional[str] = None,
    adjustments: Optional[bool] = None,
    **overrides,
) -> pd.DataFrame:
    """
    Historical Data Request (BDH).

    Parameters
    ----------
    tickers    : str ou list[str]
    flds       : str ou list[str]  (ex: 'PX_LAST' ou ['PX_LAST', 'VOLUME'])
    start_date : date de début — str 'YYYY-MM-DD',  datetime, ou pd.Timestamp
    end_date   : date de fin
    Per        : périodicité Bloomberg
                 'DAILY' | 'WEEKLY' | 'MONTHLY' | 'QUARTERLY' | 'YEARLY'
    Fill       : remplissage des jours sans cotation
                 'PREVIOUS_VALUE' | 'NIL_VALUE'
    Days       : quels jours inclure
                 'NON_TRADING_WEEKDAYS' | 'ALL_CALENDAR_DAYS' | 'ACTIVE_DAYS_ONLY'
    currency   : devise de conversion optionnelle (ex: 'EUR')
    adjust     : 'all' → ajustements dividendes + splits
                 '-'   → aucun ajustement
                 None  → comportement Bloomberg par défaut
    adjustments: alias booléen pratique
                 True  → ajustements dividendes + splits
                 False → aucun ajustement
                 None  → utilise le comportement de `adjust`
    **overrides: overrides Bloomberg supplémentaires (ex: CshAdjNormal=True)

    Returns
    -------
    pd.DataFrame
        Colonnes : MultiIndex (ticker, field) — ticker niveau 0, field niveau 1
        Index    : DatetimeIndex de nom 'date'

    Exemples
    --------
    >>> df = bdh('AAPL US Equity', 'PX_LAST', '2024-01-01', '2024-12-31')
    >>> df['AAPL US Equity']['PX_LAST']

    >>> df = bdh(['AAPL US Equity', 'MSFT US Equity'],
    ...          ['PX_LAST', 'VOLUME'],
    ...          '2024-01-01', '2024-12-31')
    >>> df['AAPL US Equity']['VOLUME']
    """
    tickers = _as_list(tickers)
    flds = _as_list(flds)
    start_str = _to_blp_date(start_date)
    end_str = _to_blp_date(end_date)
    flds_upper = [f.upper() for f in flds]

    session = get_session()
    collected: dict[str, pd.DataFrame] = {}  # ticker → DataFrame

    for batch in _chunked(tickers, CHUNK_BDH):
        req = session.create_request("HistoricalDataRequest")

        for t in batch:
            req.append("securities", t)
        for f in flds:
            req.append("fields", f)

        req.set("startDate", start_str)
        req.set("endDate", end_str)
        req.set("periodicitySelection", Per)
        req.set("nonTradingDayFillMethod", Fill)
        req.set("nonTradingDayFillOption", Days)

        if currency:
            req.set("currency", currency)

        # Gestion de l'ajustement dividendes/splits (style xbbg adjust=)
        if adjustments is True:
            req.set("adjustmentNormal", True)
            req.set("adjustmentAbnormal", True)
            req.set("adjustmentSplit", True)
        elif adjustments is False:
            req.set("adjustmentNormal", False)
            req.set("adjustmentAbnormal", False)
            req.set("adjustmentSplit", False)
        elif adjust == "all":
            req.set("adjustmentNormal", True)
            req.set("adjustmentAbnormal", True)
            req.set("adjustmentSplit", True)
        elif adjust == "-":
            req.set("adjustmentNormal", False)
            req.set("adjustmentAbnormal", False)
            req.set("adjustmentSplit", False)

        _set_overrides(req, overrides)
        session.send(req)

        for msg in _collect(session):
            sec_data = msg.getElement("securityData")
            ticker = sec_data.getElementAsString("security")
            field_data_arr = sec_data.getElement("fieldData")

            rows: list[dict] = []
            for i in range(field_data_arr.numValues()):
                fld_elem = field_data_arr.getValueAsElement(i)
                row: dict = {}
                for j in range(fld_elem.numElements()):
                    elem = fld_elem.getElement(j)
                    name = str(elem.name()).upper()
                    val = elem.getValue()
                    if val == "#N/A History":
                        val = np.nan
                    row[name] = val
                if "DATE" in row:
                    rows.append(row)

            if not rows:
                continue

            df_t = pd.DataFrame(rows)
            df_t["DATE"] = pd.to_datetime(df_t["DATE"])
            df_t = df_t.set_index("DATE")
            df_t.index.name = "date"

            # Assure la présence de toutes les colonnes demandées
            for c in flds_upper:
                if c not in df_t.columns:
                    df_t[c] = np.nan

            collected[ticker] = df_t[flds_upper]

    # DataFrame vide si aucun résultat
    if not collected:
        mi = pd.MultiIndex.from_tuples(
            [(t, f) for t in tickers for f in flds_upper],
            names=["ticker", "field"],
        )
        return pd.DataFrame(columns=mi)

    # Assemblage en MultiIndex (ticker, field)
    result = pd.concat(collected, axis=1, names=["ticker", "field"])
    result.index.name = "date"

    # Réordonne selon l'ordre original des tickers
    present = [t for t in tickers if t in collected]
    result = result.reindex(columns=present, level="ticker")

    return result


# ──────────────────────────────────────────────────────────────── BDP ─────────


def bdp(
    tickers: Union[str, List[str]],
    flds: Union[str, List[str]],
    **overrides,
) -> pd.DataFrame:
    """
    Reference Data Request — données ponctuelles (BDP).

    Parameters
    ----------
    tickers   : str ou list[str]
    flds      : str ou list[str]
    **overrides : overrides Bloomberg en **kwargs
                  ex: VWAP_Dt='20240115', END_DATE_OVERRIDE='20240101'

    Returns
    -------
    pd.DataFrame
        Index   : security names (nom 'ticker')
        Colonnes: fields demandés, noms en lowercase
                  Colonnes en erreur → NaN (sans lever d'exception)

    Exemples
    --------
    >>> bdp('AAPL US Equity', ['Security_Name', 'GICS_Sector_Name'])

    >>> bdp('AAPL US Equity', 'Eqy_Weighted_Avg_Px', VWAP_Dt='20240115')

    >>> bdp(['AAPL US Equity', 'MSFT US Equity'], 'PX_LAST')
    """
    tickers = _as_list(tickers)
    flds = _as_list(flds)
    cols_lower = [f.lower() for f in flds]

    session = get_session()
    rows: dict[str, dict] = {}

    for batch in _chunked(tickers, CHUNK_BDP):
        req = session.create_request("ReferenceDataRequest")

        for t in batch:
            req.append("securities", t)
        for f in flds:
            req.append("fields", f)

        _set_overrides(req, overrides)
        session.send(req)

        for msg in _collect(session):
            sec_arr = msg.getElement("securityData")
            for i in range(sec_arr.numValues()):
                sec = sec_arr.getValueAsElement(i)
                ticker = sec.getElementAsString("security")
                fd = sec.getElement("fieldData")

                row: dict = {}
                for j in range(fd.numElements()):
                    elem = fd.getElement(j)
                    row[str(elem.name()).lower()] = elem.getValue()

                # Champs en erreur → NaN (fieldExceptions)
                fe = sec.getElement("fieldExceptions")
                for k in range(fe.numValues()):
                    exc = fe.getValueAsElement(k)
                    fid = exc.getElementAsString("fieldId").lower()
                    row[fid] = np.nan

                rows[ticker] = row

    if not rows:
        return pd.DataFrame(
            index=pd.Index(tickers, name="ticker"),
            columns=cols_lower,
        )

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"

    # Assure la présence et l'ordre de toutes les colonnes demandées
    for c in cols_lower:
        if c not in df.columns:
            df[c] = np.nan

    return df[cols_lower]


# ──────────────────────────────────────────────────────────────── BDS ─────────


def bds(
    ticker: Union[str, List[str]],
    fld: str,
    **overrides,
) -> pd.DataFrame:
    """
    Bulk Data Request — données tabulaires (BDS).

    Parameters
    ----------
    ticker    : str ou list[str]
                Si liste, la requête est envoyée séparément pour chaque ticker
                et les résultats sont concaténés.
    fld       : champ bulk Bloomberg
                ex: 'INDX_MWEIGHT_HIST', 'INDX_MEMBERS', 'DVD_Hist_All'
    **overrides : overrides Bloomberg en **kwargs
                  ex: END_DATE_OVERRIDE='20240101', DVD_Start_Dt='20230101'

    Returns
    -------
    pd.DataFrame
        Toutes les lignes renvoyées par Bloomberg.
        Les noms de colonnes sont ceux de Bloomberg (préservés tels quels).
        Colonne 'ticker' ajoutée en tête si plusieurs tickers.

    Exemples
    --------
    >>> bds('SPX Index', 'INDX_MWEIGHT_HIST', END_DATE_OVERRIDE='20240101')

    >>> bds('AAPL US Equity', 'DVD_Hist_All',
    ...     DVD_Start_Dt='20230101', DVD_End_Dt='20231231')
    """
    if isinstance(ticker, str):
        tickers = [ticker]
        single = True
    else:
        tickers = list(ticker)
        single = len(tickers) == 1

    session = get_session()
    frames: list[pd.DataFrame] = []

    for t in tickers:
        req = session.create_request("ReferenceDataRequest")
        req.append("securities", t)
        req.append("fields", fld)

        _set_overrides(req, overrides)
        session.send(req)

        rows: list[dict] = []
        for msg in _collect(session):
            sec_arr = msg.getElement("securityData")
            for i in range(sec_arr.numValues()):
                sec = sec_arr.getValueAsElement(i)
                fd = sec.getElement("fieldData")

                if not fd.hasElement(fld):
                    continue

                bulk = fd.getElement(fld)
                for j in range(bulk.numValues()):
                    row_elem = bulk.getValueAsElement(j)
                    row: dict = {}
                    for k in range(row_elem.numElements()):
                        e = row_elem.getElement(k)
                        row[str(e.name())] = e.getValue()
                    rows.append(row)

        if rows:
            df_t = pd.DataFrame(rows)
            if not single:
                df_t.insert(0, "ticker", t)
            frames.append(df_t)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
