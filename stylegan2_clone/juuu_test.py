from django.test import TestCase
# https://www.tensorflow.org/guide/gpu
import tensorflow as tf
import pretrained_networks
from google_drive_downloader import GoogleDriveDownloader as gdd


class JuuTest:
    def __init__(self):
        pass

    def process(self):
        pass

    def downloader_pkl(self):
        url = 'https://drive.google.com/open?id=1WNQELgHnaqMTq3TlrnDaVkyrAH8Zrjez'
        #'https://drive.google.com/open?id=1BHeqOZ58WZ-vACR2MJkh1ZVbJK2B-Kle'
        model_id = url.replace('https://drive.google.com/open?id=', '')

        network_pkl = '/content/models/model_%s.pkl' % model_id#(hashlib.md5(model_id.encode()).hexdigest())
        gdd.download_file_from_google_drive(file_id=model_id,
                                            dest_path=network_pkl)

        _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
        Gs.vars["dlatent_avg"].value().eval()

if __name__ == '__main__':
    j = JuuTest()
    j.downloader_pkl()