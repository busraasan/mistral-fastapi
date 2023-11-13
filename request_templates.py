from pydantic import BaseModel, Field

class HugginfaceInferenceRequest(BaseModel):
    model_name: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.1", description="Name of the model to get inference."
    )
    prompt: str = Field(
        default="string", description="Prompt for generations"
    )
    max_length: int = Field(
        default=2048, description="Max length for generations"
    )
    do_sample: bool = Field(
        default=True
    )
    top_k: int = Field(
        default=40
    )
    top_p: float = Field(
        default=0.95
    )

    def get_generation_arguments_as_dict(self) -> dict:
        argument_dict = {}
        for k, v in self.__dict__.copy().items():
            if k not in ['prompt', 'model_name']:
                if isinstance(v, tuple):
                    v = v[0]
                if v != None:
                    argument_dict[k] = v
        return argument_dict

