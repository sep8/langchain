def create_filter(**kwargs):
    dsid = kwargs.get('dsid')
    where = kwargs.get('where', '1=1')

    return {
        "dsid": dsid,
        "where": where
    }
    