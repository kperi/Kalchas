import torch
from .dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from .model import CRNN
from .ctc_decoder import ctc_decode
from typing import List, Optional, Union
from PIL import Image

# TODO: from .factory import download_model


_models = {
    "model1": {
        "url": "kalchas/checkpoints/model1/model_best_loss_0.7971",
        "name": "model_best_loss_0.7971",
    },
    "model2": {
        "url": "kalchas/checkpoints/model2/99_model_best_loss_1.184",
        "name": "99_model_best_loss_1.184",
    },
}


def list_available_models():
    """
    Returns a list of available OCR models.

    Returns:
        list: A list of available OCR models.
    """
    return list(_models.keys())


_transform = transforms.Compose([transforms.ToTensor()])


class TextRecognizer:
    """
    A class that represents a text recognizer.

    Args:
        width (int): The width of the input image.
        height (int): The height of the input image.
        num_class (int): The number of classes for text recognition.
        model_path (str): The path to the pre-trained model.
        char2idx (dict): A dictionary mapping characters to their corresponding indices.
        idx2char (dict): A dictionary mapping indices to their corresponding characters.
        device (optional): The device to run the model on. Defaults to None.

    Attributes:
        model: The CRNN model for text recognition.
        device: The device to run the model on.
        idx2char (dict): A dictionary mapping indices to their corresponding characters.
        char2idx (dict): A dictionary mapping characters to their corresponding indices.
        width (int): The width of the input image.
        height (int): The height of the input image.

    Methods:
        ocr(images, search_strategy="beam_search", beam_size=10):
            Performs optical character recognition on the given images.

    """

    def __init__(
        self,
        width: int,
        height: int,
        num_class: int,
        model_path: str,
        char2idx: dict,
        idx2char: dict,
        device=None,
    ) -> None:
        """
        Initializes a TextRecognizer object.

        Args:
            width (int): The width of the input image.
            height (int): The height of the input image.
            num_class (int): The number of classes for text recognition.
            model_path (str): The path to the pre-trained model.
            char2idx (dict): A dictionary mapping characters to their corresponding indices.
            idx2char (dict): A dictionary mapping indices to their corresponding characters.
            device (optional): The device to run the model on. Defaults to None.
        """

        self.model = CRNN(
            1, img_height=height, img_width=width, num_class=num_class, leaky_relu=True
        ).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.device = device
        self.idx2char = idx2char
        self.char2idx = char2idx
        self.width = width
        self.height = height

    def ocr(
        self,
        images: Union[str | Image.Image, List[str] | List[Image.Image]],
        search_strategy: str = "beam_search",
        beam_size: int = 10,
        binarize=True,
    ) -> List[str] | str:
        """
        Perform optical character recognition (OCR) on the given images.

        Args:
            images (str or list[str]): The path(s) to the image(s) to perform OCR on.
            search_strategy (str, optional): The search strategy to use for decoding the predictions.
                Defaults to "beam_search".
            beam_size (int, optional): The beam size to use for beam search decoding.
                Only applicable if `search_strategy` is set to "beam_search". Defaults to 10.
            binarize (bool, optional): Whether to binarize the images before performing OCR.
                Defaults to True.

        Returns:
            list[str]: A list of predicted texts for each input image.

        """
        if not isinstance(images, list):
            images = [images]

        predict_dataset = ImageDataset(
            images,
            transform=_transform,
            width=self.width,
            height=self.height,
            has_text=False,
            char2idx=self.char2idx,
        )
        predict_dataloader = DataLoader(
            predict_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        preds = predict(
            self.model,
            self.device,
            predict_dataloader,
            label2char=self.idx2char,
            decode_method=search_strategy,
            beam_size=beam_size,
        )

        text_predictions = []
        for p in preds:
            text = "".join(p[0])
            text_predictions.append(text)

        return text_predictions

    def __repr__(self) -> str:
        return f"TextRecognizer(width={self.width}, height={self.height}, num_class={len(self.idx2char)})"


def load_ocr_model(model_name: str, device: Optional[str] = "cpu") -> TextRecognizer:
    """
    Load an OCR model based on the specified model name.

    Args:
        model_name (str): The name of the OCR model to load.
        device (str, optional): The device to load the model on (default is "cpu").

    Returns:
        model: The loaded OCR model.

    Raises:
        ValueError: If the specified model name is not found in the available models.
    """

    if model_name not in _models:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list_available_models()}"
        )

    MODEL_PATH = _models[model_name]["url"] + ".pt"
    ARTIFACT_PATH = _models[model_name]["url"] + "_artifacts.pt"

    char2idx, idx2char, width, height, MAX_TEXT_LENGTH = torch.load(ARTIFACT_PATH)
    model = TextRecognizer(
        device=device,
        width=width,
        height=height,
        num_class=len(idx2char),
        model_path=MODEL_PATH,
        idx2char=idx2char,
        char2idx=char2idx,
    )

    return model


def predict(
    model: TextRecognizer,
    device: str,
    dataloader: DataLoader,
    label2char: dict,
    decode_method: str,
    beam_size: int,
) -> List[str]:
    """
    Predicts the output for a given model using the specified decoding method.

    Args:
        model (torch.nn.Module): The trained model to use for prediction.
        device (torch.device): The device to perform the prediction on (e.g., CPU or GPU).
        dataloader (torch.utils.data.DataLoader): The data loader containing the input images.
        label2char (dict): A dictionary mapping label indices to characters.
        decode_method (str): The decoding method to use for converting logits to predictions.
        beam_size (int): The beam size for beam search decoding.
        verbose (bool, optional): Whether to print additional information during prediction. Defaults to False.

    Returns:
        list: A list of predicted outputs for each input image.

    """
    model.eval()

    all_preds = []
    with torch.no_grad():

        for data in dataloader:

            images = data["image"].to(device)
            logits = model(images)

            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            preds = ctc_decode(
                log_probs,
                method=decode_method,
                beam_size=beam_size,
                label2char=label2char,
            )
            all_preds.append(preds)

    return all_preds
