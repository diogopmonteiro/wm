from unittest import TestCase
from core.algs.cox import Cox
from core import settings
import os
import hashlib


class CoxTest(TestCase):

    def setUp(self):
        self.image_name = 'axon-lol.png'
        self.img_directory = os.path.join(settings.PROJECT_CODE_DIRECTORY, 'img')
        self.wm_img_directory = os.path.join(settings.PROJECT_CODE_DIRECTORY, 'wm-img')
        self.image_path = os.path.join(self.img_directory, self.image_name)
        self.wm_image_path = os.path.join(self.wm_img_directory, self.image_name)

    def tearDown(self):
        pass

    def _get_file_md5(self, file):
        with open(file, 'r') as f:
            m = hashlib.md5()
            m.update(f.read())
            return m.digest()

    def test_cox_diff(self):
        algorithm = Cox()
        algorithm.embed(self.image_path)

        md5_original = self._get_file_md5(self.image_path)
        md5_watermarked = self._get_file_md5(self.wm_image_path)
        self.assertNotEquals(md5_original, md5_watermarked)