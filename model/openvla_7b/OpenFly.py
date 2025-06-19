from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class OPENFLY:
    """
    OpenFly use siglip and dinov2 to process images, use llama's tokenizer to process text.
    Images will be processed by both siglip and dinov2, then the features will be concatenated and feed into a MLP Projector to map to the text embedding dimension.
    Text(Instruction) will be tokenized by llama's tokenizer, after this it's already in the text embedding dimension.
    Finally, the text and image features after MLP will be concatenated and feed into Llama 2 7b to generate the action.

    Args:
        image: PIL Image as [height, width, 3]
        instruction: Task instruction string
    """
    def __init__(self, config=None):
        from transformers import OpenVLAForActionPrediction
        self.pipeline = pipeline("image-text-to-text", model="IPEC-COMMUNITY/openfly-agent-7b")
        logger.info("Loading OpenFly model...")
        self.model = OpenVLAForActionPrediction.from_pretrained("IPEC-COMMUNITY/openfly-agent-7b")

    def forward(self, image, instruction):
        return self.model(image, instruction)