import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SSTDataset
from arguments import args
from analyzer import Analyzer

if __name__ == "__main__":

    analyzer = Analyzer(will_train=True, args=args)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(params=analyzer.model.parameters(), lr=args.lr)

    train_set = SSTDataset(
        filename="data/train.tsv",
        maxlen=args.maxlen_train,
        tokenizer=analyzer.tokenizer,
    )
    val_set = SSTDataset(
        filename="data/dev.tsv", maxlen=args.maxlen_val, tokenizer=analyzer.tokenizer
    )

    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, num_workers=args.num_threads
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads
    )


    best_accuracy = 0
    for epoch in trange(args.num_eps, desc="Epoch"):
        analyzer.train(
            train_loader=train_loader, optimizer=optimizer, criterion=criterion
        )
        val_accuracy, val_loss = analyzer.evaluate(
            val_loader=val_loader, criterion=criterion
        )
        print(
            f"Epoch {epoch} complete! Validation Accuracy : {val_accuracy}, Validation Loss : {val_loss}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(
                f"Best validation accuracy improved from {best_accuracy} to {val_accuracy}, saving analyzer..."
            )
            analyzer.save()