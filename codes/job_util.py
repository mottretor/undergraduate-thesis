from util.database_util import DatabaseUtil
from util.audio_util import AudioUtil



class Job:

    type = {"register": 1, "match": 2,"find_similar":3,"find_count":4}
    status = {"created": 1, "started": 2, "finished": 3, "errored": 4}

    def __init__(self, job_type):
        self.job_type = job_type
        db_util = DatabaseUtil()
        self.job_id = db_util.execute_insert("INSERT INTO job (type,status) VALUES ('"+str(job_type) + "','"
                                             + str(Job.status["created"]) + "')")

    def start_job(self, filepath):
        if self.job_type == Job.type["register"]:
            AudioUtil.register_songs(self.job_id)
        if self.job_type == Job.type["match"]:
            AudioUtil.match_songs(self.job_id)
        if self.job_type == Job.type["find_similar"]:
            AudioUtil.find_similar_songs(self.job_id)
        if self.job_type == Job.type["find_count"]:
            AudioUtil.find_count(self.job_id)

