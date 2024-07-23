import torch
from .dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from .model import CRNN
from .ctc_decoder import ctc_decode


_models = {
    "model1": "kalchas/artifacts/88_model_best_loss_0.7971",
}


def list_available_models():
    return [m for m in _models.keys()]


def load_ocr_model(model_name: str, device: str = "cpu"):
    if model_name not in _models:
        raise ValueError(
            f"Model {model_name} not found. Available models: {lists_models()}"
        )

    MODEL_PATH = _models[model_name] + ".pt"
    ARTIFACT_PATH = _models[model_name] + "_artifacts.pt"

    char2idx, idx2char, width, height, MAX_TEXT_LENGTH = torch.load(ARTIFACT_PATH)
    model = TextRegognizer(
        device=device,
        width=width,
        height=height,
        num_class=len(idx2char),
        model_path=MODEL_PATH,
        idx2char=idx2char,
        char2idx=char2idx,
    )

    return model


transform = transforms.Compose([transforms.ToTensor()])


def predict(
    model, device, dataloader, label2char, decode_method, beam_size, verbose=False
):
    model.eval()

    # if verbose:
    #    pbar = tqdm.tqdm(total=len(dataloader), desc="Predict")

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

            #if verbose:
            #    pbar.update(1)

        # if verbose:
        #    pbar.close()

    return all_preds


class TextRegognizer:

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
        self, images: str, search_strategy: str = "beam_search", beam_size: int = 5
    ):

        if not isinstance(images, list):
            images = [images]

        predict_dataset = ImageDataset(
            images, transform=transform, width=self.width, height=self.height, has_text=False, char2idx=self.char2idx
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
