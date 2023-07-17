from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

  
class ChartArgs(BaseModel):
    dsid: str = Field(description='This must be a data source id create by `Create Datasource` tool.')
    type: str = Field(include=['line', 'bar', 'pie'], description='Chart type')
    x_field: str = Field(title='X axis field', description='This field is used to categorize the value fields.This is a field name, you must select it from the data source fields you selected')
    y_field: str = Field(title='Y axis field', description='This field is value field that will be grouped. This is a field name, you must select it from the data source fields you selected. if the `type` is `pie`, only one `valueFields` is allowed. The value fields only support number fields.')


class ChartTool(BaseTool):
    name = "Chart"
    description = "useful for when you want to create a chart"
    args_schema: Type[ChartArgs] = ChartArgs

    def _run(
        self,
        dsid: str,
        type: str ,
        x_field,
        y_field,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return {
            'dsid': dsid,
            'type': type,
            'x_field': x_field,
            'y_field': y_field
        }

    async def _arun(
        self,
        dsid: str,
        type: str ,
        x_field,
        y_field,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Chart does not support async")