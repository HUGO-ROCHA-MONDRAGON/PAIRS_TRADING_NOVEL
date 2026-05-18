"""
Wrapper pour l'API Bloomberg (blpapi)
a la place de xbbg

Fonctions principales: bdh (historique), bdp (référence), bds (bulk data)

Usage:
    import blp_bds as blp
    df = blp.bdh(tickers=['AAPL US Equity'], flds=['PX_LAST'], 
                 start_date='2020-01-01', end_date='2020-12-31')
"""

import blpapi  # type: ignore[import-untyped]
import pandas as pd
import numpy as np
import datetime as dt


# Name objects pour parser les réponses Bloomberg
DATE = blpapi.Name("date")
ERROR_INFO = blpapi.Name("errorInfo")
EVENT_TIME = blpapi.Name("EVENT_TIME")
FIELD_DATA = blpapi.Name("fieldData")
FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
FIELD_ID = blpapi.Name("fieldId")
SECURITY = blpapi.Name("security")
SECURITY_DATA = blpapi.Name("securityData")


class BLP:
    """Classe pour gérer la connexion Bloomberg et les requêtes"""

    def __init__(self):
        self.session = blpapi.Session()

        if not self.session.start():
            raise ConnectionError("Impossible de démarrer Bloomberg")
        if not self.session.openService("//blp/refdata"):
            raise ConnectionError("Impossible d'ouvrir le service refdata")

        self.refDataSvc = self.session.getService('//blp/refdata')

    def bdh(self, strSecurity, strFields, startdate, enddate,
            per='DAILY', perAdj='CALENDAR',
            days='NON_TRADING_WEEKDAYS', fill='PREVIOUS_VALUE', curr=None):
        """Récupère l'historique de prix (BDH)"""
        request = self.refDataSvc.createRequest('HistoricalDataRequest')

        # Conversion en liste si besoin
        if isinstance(strFields, str):
            strFields = [strFields]
        if isinstance(strSecurity, str):
            strSecurity = [strSecurity]

        for security in strSecurity:
            request.append('securities', security)
        for field in strFields:
            request.append('fields', field)

        request.set('startDate', startdate.strftime('%Y%m%d'))
        request.set('endDate', enddate.strftime('%Y%m%d'))
        request.set('periodicitySelection', per)
        request.set('periodicityAdjustment', perAdj)
        request.set('nonTradingDayFillOption', days)
        request.set('nonTradingDayFillMethod', fill)

        if curr:
            request.set('currency', curr)

        self.session.sendRequest(request)

        # Récupération des données
        data = []
        keys = []

        while True:
            event = self.session.nextEvent()

            if event.eventType() not in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
                continue

            for msg in event:
                securityDataArray = msg.getElement('securityData')
                fieldData = securityDataArray.getElement('fieldData')
                fieldDataList = [fieldData.getValueAsElement(i) for i in range(fieldData.numValues())]

                df = pd.DataFrame()
                for fld in fieldDataList:
                    for v in [fld.getElement(i) for i in range(fld.numElements()) if fld.getElement(i).name() != 'date']:
                        df.loc[fld.getElementAsDatetime('date'), str(v.name())] = v.getValue()

                df.index = pd.to_datetime(df.index)
                df.replace('#N/A History', np.nan, inplace=True)

                keys.append(securityDataArray.getElementAsString('security'))
                data.append(df)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        if len(data) == 0:
            return pd.DataFrame()

        # Formatage: single ticker vs multi-ticker
        if len(strSecurity) == 1:
            data = pd.concat(data, axis=1)
            data.columns.name = 'Field'
        else:
            data = pd.concat(data, keys=keys, axis=1, names=['Security', 'Field'])
            data = data.swaplevel(axis=1)
            data = data.sort_index(axis=1, level=0)

        data.index.name = 'Date'
        return data

    def bdp(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):
        """Récupère les données de référence (BDP)"""
        request = self.refDataSvc.createRequest('ReferenceDataRequest')

        if isinstance(strFields, str):
            strFields = [strFields]
        if isinstance(strSecurity, str):
            strSecurity = [strSecurity]

        for strD in strFields:
            request.append('fields', strD)
        for strS in strSecurity:
            request.append('securities', strS)

        if strOverrideField != '':
            o = request.getElement('overrides').appendElement()
            o.setElement('fieldId', strOverrideField)
            o.setElement('value', strOverrideValue)

        self.session.sendRequest(request)

        data_dict = {}

        while True:
            event = self.session.nextEvent()

            if event.eventType() not in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
                continue

            for msg in event:
                securityDataArray = msg.getElement('securityData')

                for i in range(securityDataArray.numValues()):
                    securityData = securityDataArray.getValueAsElement(i)
                    security = securityData.getElementAsString('security')
                    fieldData = securityData.getElement('fieldData')

                    data_dict[security] = {}

                    for j in range(fieldData.numElements()):
                        field = fieldData.getElement(j)
                        data_dict[security][str(field.name())] = field.getValue()

                    # Gestion des erreurs
                    fieldExceptions = securityData.getElement('fieldExceptions')
                    for k in range(fieldExceptions.numValues()):
                        exception = fieldExceptions.getValueAsElement(k)
                        field_id = exception.getElementAsString('fieldId')
                        error_info = exception.getElement('errorInfo')
                        error_code = error_info.getElementAsInteger('errorCode')
                        error_msg = error_info.getElementAsString('message')
                        data_dict[security][field_id] = f"Error {error_code}"

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        if len(data_dict) == 0:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df.index.name = 'Security'
        return df

    def bds(self, strSecurity, strField, **overrides):
        """Récupère les données bulk (BDS) - ex: membres d'un indice"""
        request = self.refDataSvc.createRequest('ReferenceDataRequest')

        if isinstance(strSecurity, str):
            strSecurity = [strSecurity]

        for strS in strSecurity:
            request.append('securities', strS)
        request.append('fields', strField)

        # Overrides (ex: date spécifique pour historique)
        if overrides:
            ovrd_element = request.getElement('overrides')
            for key, value in overrides.items():
                o = ovrd_element.appendElement()
                o.setElement('fieldId', key)
                o.setElement('value', str(value))

        self.session.sendRequest(request)

        all_rows = []

        while True:
            event = self.session.nextEvent()

            if event.eventType() not in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
                continue

            for msg in event:
                securityDataArray = msg.getElement('securityData')

                for i in range(securityDataArray.numValues()):
                    securityData = securityDataArray.getValueAsElement(i)
                    fieldData = securityData.getElement('fieldData')

                    # On récupère les éléments de la séquence
                    if fieldData.hasElement(strField):
                        bulkElement = fieldData.getElement(strField)
                        for j in range(bulkElement.numValues()):
                            row_element = bulkElement.getValueAsElement(j)
                            row = {}
                            for k in range(row_element.numElements()):
                                elem = row_element.getElement(k)
                                row[str(elem.name())] = elem.getValue()
                            all_rows.append(row)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        if len(all_rows) == 0:
            return pd.DataFrame()

        return pd.DataFrame(all_rows)

    def closeSession(self):
        self.session.stop()


# Session globale (s'ouvre automatiquement au premier appel)
_session = None


def _get_session():
    """Crée la session Bloomberg si elle existe pas encore"""
    global _session
    if _session is None:
        _session = BLP()
    return _session


def close():
    """Ferme la session globale"""
    global _session
    if _session is not None:
        _session.closeSession()
        _session = None


# Fonctions publiques pour usage simple
# Compatible avec l'API xbbg

def bdh(tickers, flds, start_date, end_date, **kwargs):
    """Wrapper pour bdh - historique de prix"""
    s = _get_session()

    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(flds, str):
        flds = [flds]

    # Conversion string -> datetime si besoin
    if isinstance(start_date, str):
        start_dt = pd.to_datetime(start_date)
    else:
        start_dt = start_date
    if isinstance(end_date, str):
        end_dt = pd.to_datetime(end_date)
    else:
        end_dt = end_date

    raw = s.bdh(tickers, flds, start_dt, end_dt, **kwargs)

    if raw.empty:
        return raw

    # Formatage des colonnes multi-index
    if not isinstance(raw.columns, pd.MultiIndex):
        # Cas single ticker -- colonnes plates -> ajouter le niveau ticker
        raw.columns = pd.MultiIndex.from_product([tickers, raw.columns])
    else:
        # On swap pour avoir (ticker, field)
        raw = raw.swaplevel(axis=1)
        raw = raw.sort_index(axis=1, level=0)

    return raw


def bdp(tickers, flds, **kwargs):
    """Wrapper pour bdp - données de référence"""
    s = _get_session()
    return s.bdp(tickers, flds)


def bds(tickers, flds, **kwargs):
    """Wrapper pour bds - données bulk"""
    s = _get_session()
    return s.bds(tickers, flds, **kwargs)