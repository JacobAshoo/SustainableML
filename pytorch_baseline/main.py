if __name__ == "__main__":
    import torch, torch.nn as nn, torch.optim as optim
    import time, json

    from config import *
    from system import hw, rss, mb
    from data import get_cifar10
    from model import Net
    from engine import train, evaluate
    from tracking import start, stop
if __name__ == "__main__":
    import torch, torch.nn as nn, torch.optim as optim
    import time, json

    from config import *
    from system import hw, rss, mb
    from data import get_cifar10
    from model import Net
    from engine import train, evaluate
    from tracking import start, stop
    from metrics import write

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_dl, test_dl = get_cifar10(BATCH, NW)

    model = Net().to(DEV)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    cpu_peak = rss()
    def bump():
        global cpu_peak
        cpu_peak = max(cpu_peak, rss())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    tracker = start("baseline_clean")

    train_acc, train_s = train(model, train_dl, opt, loss_fn, EPOCHS, DEV, bump)
    test_acc, eval_s  = evaluate(model, test_dl, DEV, bump)

    co2 = stop(tracker)

    row = {
     "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "variant":"baseline_clean",
     "dataset":"CIFAR-10",
     "model":"3Conv+GAP+Linear",
     "epochs":EPOCHS,
     "batch_size":BATCH,
     "lr":LR,
     "seed":SEED,
     "train_runtime_s":round(train_s,4),
     "eval_runtime_s":round(eval_s,4),
     "train_accuracy":round(train_acc,4),
     "test_accuracy":round(float(test_acc),4),
     "cpu_peak_rss_mb":mb(cpu_peak),
     "hw_info_json":json.dumps(hw(DEV))
    }

    write(row, OUT_CSV)
    from metrics import write

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_dl, test_dl = get_cifar10(BATCH, NW)

    model = Net().to(DEV)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    cpu_peak = rss()
    def bump():
        global cpu_peak
        cpu_peak = max(cpu_peak, rss())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    tracker = start("baseline_clean")
if __name__ == "__main__":
    import torch, torch.nn as nn, torch.optim as optim
    import time, json

    from config import *
    from system import hw, rss, mb
    from data import get_cifar10
    from model import Net
    from engine import train, evaluate
    from tracking import start, stop
    from metrics import write

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_dl, test_dl = get_cifar10(BATCH, NW)

    model = Net().to(DEV)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    cpu_peak = rss()
    def bump():
        global cpu_peak
        cpu_peak = max(cpu_peak, rss())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    tracker = start("baseline_clean")

    train_acc, train_s = train(model, train_dl, opt, loss_fn, EPOCHS, DEV, bump)
    test_acc, eval_s  = evaluate(model, test_dl, DEV, bump)

    co2 = stop(tracker)

    row = {
     "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "variant":"baseline_clean",
     "dataset":"CIFAR-10",
     "model":"3Conv+GAP+Linear",
     "epochs":EPOCHS,
     "batch_size":BATCH,
     "lr":LR,
     "seed":SEED,
     "train_runtime_s":round(train_s,4),
     "eval_runtime_s":round(eval_s,4),
     "train_accuracy":round(train_acc,4),
     "test_accuracy":round(float(test_acc),4),
     "cpu_peak_rss_mb":mb(cpu_peak),
     "hw_info_json":json.dumps(hw(DEV))
    }

    write(row, OUT_CSV)

    train_acc, train_s = train(model, train_dl, opt, loss_fn, EPOCHS, DEV, bump)
    test_acc, eval_s  = evaluate(model, test_dl, DEV, bump)

    co2 = stop(tracker)

    row = {
     "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "variant":"baseline_clean",
     "dataset":"CIFAR-10",
     "model":"3Conv+GAP+Linear",
     "epochs":EPOCHS,
     "batch_size":BATCH,
     "lr":LR,
     "seed":SEED,
     "train_runtime_s":round(train_s,4),
     "eval_runtime_s":round(eval_s,4),
     "train_accuracy":round(train_acc,4),
     "test_accuracy":round(float(test_acc),4),
     "cpu_peak_rss_mb":mb(cpu_peak),
     "hw_info_json":json.dumps(hw(DEV))
    }

    write(row, OUT_CSV)

