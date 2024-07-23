import torch
from .dataset import ImageDataset
from torch.utils.data import DataLoader
import tqdm
from torchvision import transforms
from .model import CRNN
from .ctc_decoder import ctc_decode

transform = transforms.Compose([transforms.ToTensor()])


def predict(
    model, device, dataloader, label2char, decode_method, beam_size, verbose=False
):
    model.eval()

    if verbose:
        pbar = tqdm.tqdm(total=len(dataloader), desc="Predict")

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

            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

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

    def ocr(self, images, search_strategy="beam_search", beam_size=5):

        if not isinstance(images, list):
            images = [images]

        predict_dataset = ImageDataset(
            images, transform=transform, has_text=False, char2idx=self.char2idx
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
