from typing import Tuple, Union

from transformers import AutoProcessor, LlamaForConditionalGeneration

from swarms.models.base_multimodal_model import BaseMultiModalModel


class LlamaMultiModal(BaseMultiModalModel):
    """
    A class to handle text inputs using the Meta-Llama-3.1-8B-Instruct model for conditional generation.

    Attributes:
        model_name (str): The name or path of the pre-trained model.
        max_length (int): The maximum length of the generated sequence.

    Args:
        model_name (str): The name of the pre-trained model.
        max_length (int): The maximum length of the generated sequence.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Examples:
    >>> model = LlamaMultiModal()
    >>> model.run("Tell me a story about a pirate.")

    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_length: int = 256,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.max_length = max_length

        self.model = LlamaForConditionalGeneration.from_pretrained(
            model_name, *args, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def run(
        self, text: str, *args, **kwargs
    ) -> Union[str, Tuple[None, str]]:
        """
        Processes the input text and generates a response.

        Args:
            text (str): The input text for the model.

        Returns:
            Union[str, Tuple[None, str]]: The generated response string or a tuple (None, error message) in case of an error.
        """
        try:
            inputs = self.processor(
                text=text, return_tensors="pt"
            )

            # Generate
            generate_ids = self.model.generate(
                **inputs, max_length=self.max_length, **kwargs
            )
            return self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                *args,
            )[0]

        except Exception as e:
            return None, f"Error during model processing: {str(e)}"