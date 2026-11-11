"""No implicit dataset, network access, or GT dependency."""
class LocalRaw:
    def __init__(self):raise ValueError('An explicit raw .npy/.tif path is required')
    def prefetch(self,keys,wait=True):return []
