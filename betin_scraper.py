from bs4 import BeautifulSoup
import re
import sys
import time
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import Select

class Scrapper():

	def __init__(self):
		self.browser = webdriver.Chrome(
			"C:/Users/Naomi/Desktop/personal/Locker/ML_project/research/epl-prediction-2017/"+
			"console/chromedriver/chromedriver.exe")
		self.get_betin()

	def get_betin(self):
		self.login_betin()
		self.navigate_to_match()

	def login_betin(self):
		self.browser.get("https://sports.betin.co.ke/mobile#/login")
		print("Getting betin...")
		self.timer(2)
		print("Login in...")
		user = self.browser.find_element_by_css_selector(
			"#wrapper > main > div > div > form > input.credentials__input.mt20")
		user.send_keys("0706033970")
		password = self.browser.find_element_by_css_selector(
			"#wrapper > main > div > div > form > input.credentials__input.mt15")
		password.send_keys("ABCabc123")
		self.timer(1)
		self.browser.find_element_by_css_selector(
			"#wrapper > main > div > div > form > button").click()
		print("Loged in successfully...")
		self.timer(2)

	def navigate_to_match(self):
		self.browser.get("https://sports.betin.co.ke/mobile#/eventsInCoupon/eurolist/1-3")
		self.timer(5)
		self.browser.find_element_by_css_selector(
			"#wrapper > main > div > div:nth-child(6) > div.accordion-toggle.table-f").click()

	def timer(self, t):
		while t:
			mins, secs = divmod(t, 60)
			timeformat = '{:02d}:{:02d}'.format(mins, secs)
			print(timeformat, end='\r')
			time.sleep(1)
			t -= 1

	def scrape(self):
		odds_tbl = self.browser.find_element_by_css_selector(
			"#wrapper > main > div > div:nth-child(6) > div.accordion-content")
		odds_tbl_html = odds_tbl.get_attribute("innerHTML")
		odds_tbl_soup = BeautifulSoup(odds_tbl_html, "html.parser")
		#todo.....
		print()

if __name__ == '__main__':
	Scrapper()