from util.job_util import Job

if __name__ == '__main__':
    match_job = Job(Job.type["match"])
    print("Job Started. Job ID : " + str(match_job.job_id))
    match_job.start_job(None)







