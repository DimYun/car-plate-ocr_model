"""Model construction."""
import torch
from timm import create_model


class CRNN(torch.nn.Module):
    """CRNN model for OCR task.

    CNN-backbone from timm, in RNN part used GRU.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        cnn_output_size: int = 128,
        rnn_features_num: int = 48,
        rnn_hidden_size: int = 64,
        rnn_dropout: float = 0.1,
        rnn_bidirectional: bool = True,
        rnn_num_layers: int = 2,
        num_classes: int = 11,
    ) -> None:
        super().__init__()

        # Pretrained backbone for features.
        # Can be cut, don't necessary use whole depth.
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2,),
        )

        self.gate = torch.nn.Conv2d(
            cnn_output_size, rnn_features_num, kernel_size=1, bias=False
        )

        # Recurrent part.
        self.rnn = torch.nn.GRU(
            input_size=384,  # 576,
            hidden_size=rnn_hidden_size,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            num_layers=rnn_num_layers,
        )

        classifier_in_features = rnn_hidden_size
        if rnn_bidirectional:
            classifier_in_features = 2 * rnn_hidden_size

        # Classificator.
        self.fc = torch.nn.Linear(classifier_in_features, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Function for make prediction with model.

        :param tensor: input tensor to predict
        :return: predicted values
        """
        cnn_features = self.backbone(tensor)[0]
        cnn_features = self.gate(cnn_features)
        cnn_features = cnn_features.permute(3, 0, 2, 1)
        cnn_features = cnn_features.reshape(
            cnn_features.shape[0],
            cnn_features.shape[1],
            cnn_features.shape[2] * cnn_features.shape[3],
        )
        rnn_output, _ = self.rnn(cnn_features)
        logits = self.fc(rnn_output)
        return self.softmax(logits)
