
import unittest
from dragnet import blocks

class Textencoding(unittest.TestCase):
    def test_guess_encoding(self):
        s = """<?xml version="1.0" encoding="ISO-8859-1"?>
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
          "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">

          <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="fr" lang="fr">
          """
        self.assertEqual(blocks.guess_encoding(s), 'ISO-8859-1')

        s = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"
          "http://www.w3.org/TR/html4/strict.dtd">
                 
          <head>
          <meta http-equiv="content-type" content="text/html; charset=GB2312">
          </head>"""
        self.assertEqual(blocks.guess_encoding(s), 'GB2312')

        s = """<html>sadfsa</html>"""
        self.assertEqual(blocks.guess_encoding(s, 'asciI'), 'asciI')


if __name__ == "__main__":
    unittest.main()

