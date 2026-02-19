import platform, psutil, cpuinfo, os, torch

def hw(device):
  d = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "device": device,
    "ram_gb": round(psutil.virtual_memory().total/(1024**3),2)
  }
  try:
    d["cpu"] = cpuinfo.get_cpu_info().get("brand_raw","")
  except:
    d["cpu"] = ""

  if torch.cuda.is_available():
    d["gpu"] = torch.cuda.get_device_name(0)
    d["cuda"] = torch.version.cuda
  else:
    d["gpu"] = ""
    d["cuda"] = ""

  return d

def rss():
  return psutil.Process(os.getpid()).memory_info().rss

def mb(x):
  return round(x/(1024**2),2)
