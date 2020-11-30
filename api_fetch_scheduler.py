import pairs_info
import monitor_pairs

import os
pid = os.getpid()
op = open("api_monitor.pid","w")
op.write("%s" % pid)
op.close()

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('keep', default=False, type=bool, help='keep the process waiting for interrupt from user')

args = parser.parse_args()
_keep  = args.keep



pairs_info=pairs_info.get_pairs_info()
all_pairs_list=pairs_info["all_pairs_list"]
all_intervals_list=pairs_info["all_intervals_list"]
available_pairs_df = pairs_info["available_pairs_df"]


#from apscheduler.scheduler import Scheduler
from apscheduler.schedulers.background import BackgroundScheduler
import logging
logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.DEBUG)
import requests

# Start the scheduler
#sched = Scheduler()
#sched.start()

def hourly_job():
        import monitor_pairs
        monitor_pairs.pairs2csv(pairs=all_pairs_list,intervals=[1]) #2m when no file exists, 30 sec if all files exists
def daily_job():
        import monitor_pairs
        monitor_pairs.pairs2csv(pairs=all_pairs_list,intervals=[15])
def weekly_job():
        import monitor_pairs
        monitor_pairs.pairs2csv(pairs=all_pairs_list,intervals=[1440]) #30 sec if all files exists
        monitor_pairs.pairs2csv(pairs=all_pairs_list,intervals=[21600])



# Schedules job_function to be run on the third Friday
# of June, July, August, November and December at 00:00, 01:00, 02:00 and 03:00
#sched.add_cron_job(job_function, month='6-8,11-12', day='3rd fri', hour='0-3')
#sched.add_cron_job(job_function, month='*', day='*', hour='21',minute='14')


scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(hourly_job, trigger='cron', minute=10,id='hourly_job')
scheduler.add_job(daily_job, trigger='cron', hour=4,id='daily_job')
scheduler.add_job(weekly_job, trigger='cron', day_of_week='2',id='weekly_job')
print('here is the schedule')
scheduler.print_jobs()

if _keep:
	input("Press CTLR+C to exit \n\n")

