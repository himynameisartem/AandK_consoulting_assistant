import phoenix as px
import phoenix.otel
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor


def launch_phoenix():
    session = px.launch_app()
    phoenix.otel.register(
        project_name="a-and-k-consulting-assistant",
        endpoint="http://127.0.0.1:6006/v1/traces",
        protocol="http/protobuf",
    )
    LangChainInstrumentor().instrument()
    OpenAIInstrumentor().instrument()
    return session
