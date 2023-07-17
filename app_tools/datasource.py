import json
from uuid import UUID
from pydantic import BaseModel, Field
from langchain.tools.base import Tool
import requests

supported_types = ['esriFieldTypeString', 'esriFieldTypeDouble',
                   'esriFieldTypeInteger', 'esriFieldTypeSmallInteger']
filtered_filed_names = ['Shape', 'Shape_Length', 'Shape_Area']
def get_service_schema(service): 
    response = requests.get(service+'?f=json')
    definition = json.loads(response.text)
    name = definition['name']
    fields = definition['fields']
    fields = [field for field in fields if field['type']
            in supported_types and field['name'] not in filtered_filed_names]
    fields = [{"name": field['name'], "alias": field['alias']
            or field['name'], "type": field['type']} for field in fields]
    return {"name": name, "fields": fields}

service = 'https://sampleserver6.arcgisonline.com/arcgis/rest/services/Census/MapServer/3'

def create_datasource(keys):
    uuid = UUID('{12345678-1234-5678-1234-567812345678}')
    schema = get_service_schema(service)
    return {
        "type": 'datasource',
        "id": f"{''.join(keys)}{uuid}",
        "name": schema['name'],
        "fields": schema['fields'][:12]
    }

class DataSourceArgs(BaseModel):
    keys: list = Field(description='Some keywords used to create data sources')


DataSourceTool = Tool.from_function(
    func=create_datasource,
    name="Datasource",
    description="useful for when you want to create a data source with keywords",
    args_schema=DataSourceArgs
)