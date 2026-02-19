import torch, time

def train(model, dl, opt, loss_fn, epochs, device, bump):

  model.train()
  correct = 0
  total = 0

  start = time.perf_counter()

  for _ in range(epochs):
    for x,y in dl:
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      opt.zero_grad(set_to_none=True)
      out = model(x)
      loss = loss_fn(out,y)
      loss.backward()
      opt.step()

      pred = out.argmax(1)
      correct += (pred==y).sum().item()
      total += y.size(0)
      bump()

  return correct/total, time.perf_counter()-start


@torch.no_grad()
def evaluate(model, dl, device, bump):

  model.eval()
  c,n = 0,0

  start = time.perf_counter()

  for x,y in dl:
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    out = model(x)
    p = out.argmax(1)
    c += (p==y).sum().item()
    n += y.size(0)
    bump()

  return c/n, time.perf_counter()-start
