def create_table(**kwargs):
    dsid = kwargs.get('dsid')
    name = kwargs.get('name', 'Table 1')

    return {
        "dsid": dsid,
        "name": name
    }
    