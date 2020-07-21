from util.job_util import Job

if __name__ == '__main__':
    register_job = Job(Job.type["register"])
    print("Job Started. Job ID : " + str(register_job.job_id))
    register_job.start_job(None)