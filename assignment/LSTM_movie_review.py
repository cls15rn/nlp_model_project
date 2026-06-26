# -*- coding: utf-8 -*-
"""
네이버 영화 리뷰(NSMC) 감성 분석 LSTM 모델 - PyTorch Lightning 버전

기존 lstm_imdb_lightning.py 를 커스터마이징한 파일입니다.
주요 변경점:
  1. 데이터: IMDB(영어, 폴더 구조) -> NSMC ratings.txt(한글, 탭 구분 TSV)
  2. 전처리: 영어 정규식/소문자 -> 한글 정규식 + (선택) KoNLPy 형태소 분석
  3. 데이터 분할: 하나의 ratings.txt 를 train/test 로 직접 분할
  4. 예제(toy) 데이터와 예측 예시 문장을 한글로 교체
모델 구조(Embedding + LSTM + Linear)와 Lightning 학습 흐름은 원본과 동일합니다.
"""

# ---------------------------------------------------------------------
# 1. 기본 라이브러리 불러오기
# ---------------------------------------------------------------------

# os는 폴더 생성, 파일 경로 확인, 디렉터리 탐색 등에 사용합니다.
import os

# re는 정규표현식을 사용하여 특수문자 제거 등을 처리할 때 사용합니다.
import re

# random은 데이터를 섞거나 train/test로 나눌 때 사용합니다.
import random

# Counter는 단어가 몇 번 등장했는지 세어 vocabulary를 만들 때 사용합니다.
from collections import Counter

# dataclass는 설정값을 하나의 객체로 깔끔하게 묶기 위해 사용합니다.
from dataclasses import dataclass, field

# Path는 Windows와 macOS/Linux 경로를 안전하게 다루기 위해 사용합니다.
from pathlib import Path

# typing은 함수 인자와 반환값의 타입을 명확하게 표시하기 위해 사용합니다.
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# 2. 딥러닝 라이브러리 불러오기
# ---------------------------------------------------------------------

# torch는 PyTorch의 핵심 라이브러리입니다.
import torch

# nn은 Embedding, LSTM, Linear, Dropout 같은 신경망 계층을 제공합니다.
import torch.nn as nn

# Dataset과 DataLoader는 데이터를 모델에 배치 단위로 공급하기 위해 사용합니다.
from torch.utils.data import Dataset, DataLoader

# random_split은 하나의 훈련 데이터를 훈련/검증 데이터로 나누기 위해 사용합니다.
from torch.utils.data import random_split

# PyTorch Lightning은 학습 루프를 구조적으로 관리하는 라이브러리입니다.
import pytorch_lightning as pl

# torchmetrics는 정확도 같은 평가 지표를 안정적으로 계산하기 위해 사용합니다.
from torchmetrics.classification import BinaryAccuracy


# ---------------------------------------------------------------------
# 3. 설정값 정의
# ---------------------------------------------------------------------

def _default_data_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "data" / "ratings.txt")


@dataclass
class Config:
    data_path: str = field(default_factory=_default_data_path)

    encoding: str = "utf-8"

    max_len: int = 40

    max_vocab_size: int = 20000

    min_freq: int = 2

    batch_size: int = 64

    embedding_dim: int = 128

    hidden_dim: int = 128

    num_layers: int = 1

    dropout: float = 0.3

    learning_rate: float = 0.001

    max_epochs: int = 3

    test_ratio: float = 0.2

    val_ratio: float = 0.2

    use_konlpy: bool = True

    num_workers: int = 0

    seed: int = 42

    use_toy_data_if_load_fails: bool = True


# ---------------------------------------------------------------------
# 4. 텍스트 전처리 함수 (★ 한글용으로 수정된 핵심 부분)
# ---------------------------------------------------------------------

STOPWORDS = {
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과",
    "도", "으로", "로", "에서", "하다", "있다", "되다", "그", "저", "것",
}


def clean_text(text: str) -> str:

    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9 ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

_okt = None
_konlpy_checked = False


def _get_okt(use_konlpy: bool):

    global _okt, _konlpy_checked

    if not use_konlpy:
        return None

    if _konlpy_checked:
        return _okt

    _konlpy_checked = True
    try:
        from konlpy.tag import Okt
        _okt = Okt()
        print("[전처리] KoNLPy Okt 형태소 분석기를 사용합니다.")
    except Exception as error:
        _okt = None
        print(f"[전처리] KoNLPy를 사용할 수 없어 공백 기준 토큰화로 대체합니다. ({error})")

    return _okt


def tokenize(text: str, use_konlpy: bool = True) -> List[str]:

    cleaned = clean_text(text)

    okt = _get_okt(use_konlpy)

    if okt is not None:
        tokens = okt.morphs(cleaned, stem=True)
    else:
        tokens = cleaned.split()

    return [t for t in tokens if t and t not in STOPWORDS]


# ---------------------------------------------------------------------
# 5. NSMC 데이터 로드 함수 (★ IMDB 다운로드/폴더 읽기를 대체)
# ---------------------------------------------------------------------

def read_nsmc_file(path: Path, encoding: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []

    with open(path, "r", encoding=encoding) as f:

        header = f.readline()

        for line in f:

            parts = line.rstrip("\n").split("\t")

            if len(parts) != 3:
                continue

            _id, document, label = parts

            if document.strip() == "":
                continue

            try:
                label_id = int(label)
            except ValueError:
                continue

            samples.append((document, label_id))

    return samples


def make_toy_samples() -> List[Tuple[str, int]]:

    positive = [
        "정말 재미있고 감동적인 영화였어요",
        "배우들 연기가 훌륭하고 스토리가 탄탄합니다",
        "올해 본 영화 중에 최고였다 강력 추천",
        "음악도 좋고 영상미가 아름다운 작품",
        "시간 가는 줄 모르고 몰입해서 봤어요",
        "결말이 너무 만족스럽고 여운이 길게 남는다",
    ]

    negative = [
        "너무 지루하고 시간이 아까운 영화였다",
        "연기도 어색하고 스토리가 엉망이에요",
        "기대했는데 정말 실망스러운 작품",
        "내용이 뻔하고 전개가 답답합니다",
        "별점 주기도 아까운 최악의 영화",
        "끝까지 보기 힘들 정도로 재미없었다",
    ]

    samples = [(t, 1) for t in positive] + [(t, 0) for t in negative]

    samples = samples * 30

    return samples


def load_data(config: Config) -> List[Tuple[str, int]]:

    data_path = Path(config.data_path)

    try:
        if not data_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path.resolve()}")

        samples = read_nsmc_file(data_path, config.encoding)

        print(f"[데이터 로드 완료] 전체 리뷰 수: {len(samples)}")
        return samples

    except Exception as error:
        print(f"[경고] NSMC 데이터 로드 실패: {error}")

        if not config.use_toy_data_if_load_fails:
            raise

        print("[대체 실행] ratings.txt 로드가 불가능하여 작은 예제 데이터로 실행합니다.")
        return make_toy_samples()


def split_train_test(
    samples: List[Tuple[str, int]], test_ratio: float, seed: int
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    test_size = int(len(shuffled) * test_ratio)

    test_samples = shuffled[:test_size]
    train_samples = shuffled[test_size:]

    print(f"[데이터 분할] train={len(train_samples)}, test={len(test_samples)}")
    return train_samples, test_samples


# ---------------------------------------------------------------------
# 6. Vocabulary 생성 함수
# ---------------------------------------------------------------------

def build_vocab(samples: List[Tuple[str, int]], config: Config) -> Dict[str, int]:

    counter: Counter = Counter()

    for text, _ in samples:
        counter.update(tokenize(text, config.use_konlpy))

    word_to_index: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    for word, freq in counter.most_common(config.max_vocab_size - len(word_to_index)):

        if freq < config.min_freq:
            continue

        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

    print(f"[Vocabulary 생성 완료] 단어 수: {len(word_to_index)}")
    return word_to_index


def encode_text(text: str, word_to_index: Dict[str, int], config: Config) -> torch.Tensor:

    tokens = tokenize(text, config.use_konlpy)

    token_ids = [word_to_index.get(tok, word_to_index["<unk>"]) for tok in tokens]

    token_ids = token_ids[: config.max_len]

    if len(token_ids) < config.max_len:
        token_ids = token_ids + [word_to_index["<pad>"]] * (config.max_len - len(token_ids))

    return torch.tensor(token_ids, dtype=torch.long)


# ---------------------------------------------------------------------
# 7. Dataset 클래스 정의
# ---------------------------------------------------------------------

class NSMCDataset(Dataset):

    def __init__(self, samples: List[Tuple[str, int]], word_to_index: Dict[str, int], config: Config):
        self.samples = samples

        self.word_to_index = word_to_index

        self.config = config

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text, label = self.samples[index]

        input_ids = encode_text(text, self.word_to_index, self.config)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_ids, label_tensor


# ---------------------------------------------------------------------
# 8. LightningDataModule 정의
# ---------------------------------------------------------------------

class NSMCDataModule(pl.LightningDataModule):

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.word_to_index: Dict[str, int] = {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        all_samples = load_data(self.config)

        train_samples, test_samples = split_train_test(
            all_samples, self.config.test_ratio, self.config.seed
        )

        self.word_to_index = build_vocab(train_samples, self.config)

        full_train_dataset = NSMCDataset(train_samples, self.word_to_index, self.config)
        self.test_dataset = NSMCDataset(test_samples, self.word_to_index, self.config)

        val_size = int(len(full_train_dataset) * self.config.val_ratio)
        train_size = len(full_train_dataset) - val_size

        generator = torch.Generator().manual_seed(self.config.seed)

        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator,
        )

        print(
            f"[Dataset 준비 완료] train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


# ---------------------------------------------------------------------
# 9. LSTM 모델 정의 (원본과 동일)
# ---------------------------------------------------------------------

class LSTMClassifier(pl.LightningModule):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        pad_index: int = 0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_dim, 2)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        embedded = self.embedding(input_ids)

        output, (hidden, cell) = self.lstm(embedded)

        sentence_vector = hidden[-1]

        sentence_vector = self.dropout(sentence_vector)

        logits = self.classifier(sentence_vector)

        return logits

    def _shared_step(self, batch, stage: str):
        input_ids, labels = batch

        logits = self(input_ids)

        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            acc = self.train_acc(preds, labels)
        elif stage == "val":
            acc = self.val_acc(preds, labels)
        else:
            acc = self.test_acc(preds, labels)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Adam optimizer를 생성합니다.
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# ---------------------------------------------------------------------
# 10. 예측 함수
# ---------------------------------------------------------------------

def predict_sentiment(
    model: LSTMClassifier, text: str, word_to_index: Dict[str, int], config: Config
) -> Tuple[str, float]:

    model.eval()

    with torch.no_grad():

        input_ids = encode_text(text, word_to_index, config).unsqueeze(0)

        input_ids = input_ids.to(model.device)

        logits = model(input_ids)

        probabilities = torch.softmax(logits, dim=1)

        pred_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_id].item()

    label = "긍정" if pred_id == 1 else "부정"

    return label, confidence


# ---------------------------------------------------------------------
# 11. main 함수
# ---------------------------------------------------------------------

def main() -> None:

    config = Config()

    pl.seed_everything(config.seed, workers=True)

    data_module = NSMCDataModule(config)

    data_module.setup(stage="fit")

    vocab_size = len(data_module.word_to_index)

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        pad_index=data_module.word_to_index["<pad>"],
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)

    examples = [
        "이 영화 정말 재미있고 감동적이었어요 강력 추천합니다",
        "시간이 너무 아깝고 지루했던 최악의 영화",
    ]

    print("\n[예측 예시]")
    for text in examples:
        label, confidence = predict_sentiment(model, text, data_module.word_to_index, config)
        print(f"문장: {text}")
        print(f"예측: {label}, 신뢰도: {confidence:.4f}\n")


# ---------------------------------------------------------------------
# 12. 프로그램 시작 지점
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()