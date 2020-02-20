import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2 as cv
from util.database_util import DatabaseUtil
from multiprocessing import Pool
import os.path
import gc
import datetime


class AudioUtil:

    @staticmethod
    def find_count(job_id):
        gc.enable()
        db_util = DatabaseUtil()
        song_list = db_util.execute_query("SELECT id FROM song where id<'10'")
        for (query_id,) in song_list:
            for des_count in range(19):
                des_name = "des_"+str(des_count)
                des_node = db_util.execute_query("SELECT "+des_name+" FROM song where id='"+str(query_id)+"'")
                for (query_descriptor,) in des_node:
                    query_descriptor = np.frombuffer(query_descriptor, dtype=np.float32)
                    query_descriptor = query_descriptor.reshape(-1, 128)
                    keypoint_count = query_descriptor.shape[0]
                    print(str(query_id)+" : "+str(query_id)+" : "+str(keypoint_count))
                    db_util.execute_update("INSERT INTO keypoint_count VALUES ('"+str(query_id)+"','"+str(des_count)+"','"+str(keypoint_count)+"')")
        db_util.execute_update("UPDATE job SET status='3' WHERE id='" + str(job_id) + "'")
        db_util.close()
        gc.collect()

    @staticmethod
    def match_songs(job_id):
        gc.enable()

        db_util = DatabaseUtil()

        song_ids = db_util.execute_query("SELECT id FROM song where id > '1988'")
        for (song_id,) in song_ids:
            AudioUtil.match_song(song_id)
            print("Song : "+str(song_id)+" Done")
        db_util.execute_update("UPDATE job SET status='3' WHERE id='" + str(job_id) + "'")
        db_util.close()
        gc.collect()

    @staticmethod
    def find_similar_songs(job_id):
        db_util = DatabaseUtil()
        bf = cv.BFMatcher()
        song_list = db_util.execute_query("SELECT id,des_0 FROM song")
        for (query_id, query_descriptor) in song_list:
            query_descriptor = np.frombuffer(query_descriptor, dtype=np.float32)
            query_descriptor = query_descriptor.reshape(-1, 128)
            for (list_id, list_descriptor) in song_list:
                list_descriptor = np.frombuffer(list_descriptor, dtype=np.float32)
                list_descriptor = list_descriptor.reshape(-1, 128)
                matches = bf.knnMatch(list_descriptor, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if (count > 500) and (query_id!=list_id):
                    db_util.execute_insert("INSERT INTO same_song (song_id, same_id) VALUES ('"+str(query_id)+"','"+str(list_id)+"')")
        db_util.execute_update("UPDATE job SET status='3' WHERE id='" + str(job_id) + "'")
        db_util.close()
        gc.collect()



    @staticmethod
    def match_song(song_id):
        db_util = DatabaseUtil()
        bf = cv.BFMatcher()
        song_list = db_util.execute_query("SELECT id,des_0 FROM song where id < '1989'")
        descriptor_list = db_util.execute_query("SELECT * FROM song WHERE id='"+str(song_id)+"'")
        for (id, title, des_0, des_1, des_2, des_3, des_4, des_5, des_6, des_7, des_8, des_9, des_10, des_11, des_12,des_13,des_14,des_15,des_16,des_17,des_18) in descriptor_list:
            best_song = [0]*19
            best_count = [0]*19

            des_0 = np.frombuffer(des_0, dtype=np.float32)
            des_0 = des_0.reshape(-1, 128)

            des_1 = np.frombuffer(des_1, dtype=np.float32)
            des_1 = des_1.reshape(-1, 128)

            des_2 = np.frombuffer(des_2, dtype=np.float32)
            des_2 = des_2.reshape(-1, 128)

            des_3 = np.frombuffer(des_3, dtype=np.float32)
            des_3 = des_3.reshape(-1, 128)

            des_4 = np.frombuffer(des_4, dtype=np.float32)
            des_4 = des_4.reshape(-1, 128)

            des_5 = np.frombuffer(des_5, dtype=np.float32)
            des_5 = des_5.reshape(-1, 128)

            des_6 = np.frombuffer(des_6, dtype=np.float32)
            des_6 = des_6.reshape(-1, 128)

            des_7 = np.frombuffer(des_7, dtype=np.float32)
            des_7 = des_7.reshape(-1, 128)

            des_8 = np.frombuffer(des_8, dtype=np.float32)
            des_8 = des_8.reshape(-1, 128)

            des_9 = np.frombuffer(des_9, dtype=np.float32)
            des_9 = des_9.reshape(-1, 128)

            des_10 = np.frombuffer(des_10, dtype=np.float32)
            des_10 = des_10.reshape(-1, 128)

            des_11 = np.frombuffer(des_11, dtype=np.float32)
            des_11 = des_11.reshape(-1, 128)

            des_12 = np.frombuffer(des_12, dtype=np.float32)
            des_12 = des_12.reshape(-1, 128)

            des_13 = np.frombuffer(des_13, dtype=np.float32)
            des_13 = des_13.reshape(-1, 128)

            des_14 = np.frombuffer(des_14, dtype=np.float32)
            des_14 = des_14.reshape(-1, 128)

            des_15 = np.frombuffer(des_15, dtype=np.float32)
            des_15 = des_15.reshape(-1, 128)

            des_16 = np.frombuffer(des_16, dtype=np.float32)
            des_16 = des_16.reshape(-1, 128)

            des_17 = np.frombuffer(des_17, dtype=np.float32)
            des_17 = des_17.reshape(-1, 128)

            des_18 = np.frombuffer(des_18, dtype=np.float32)
            des_18 = des_18.reshape(-1, 128)

            for (query_id,query_descriptor) in song_list:

                query_descriptor = np.frombuffer(query_descriptor, dtype=np.float32)
                query_descriptor = query_descriptor.reshape(-1, 128)

                # Descriptor 0
                matches = bf.knnMatch(des_0, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[0]:
                    best_count[0] = count
                    best_song[0] = query_id

                # Descriptor 1
                matches = bf.knnMatch(des_1, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[1]:
                    best_count[1] = count
                    best_song[1] = query_id

                # Descriptor 2
                matches = bf.knnMatch(des_2, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[2]:
                    best_count[2] = count
                    best_song[2] = query_id

                # Descriptor 3
                matches = bf.knnMatch(des_3, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[3]:
                    best_count[3] = count
                    best_song[3] = query_id

                # Descriptor 4
                matches = bf.knnMatch(des_4, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[4]:
                    best_count[4] = count
                    best_song[4] = query_id

                # Descriptor 5
                matches = bf.knnMatch(des_5, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[5]:
                    best_count[5] = count
                    best_song[5] = query_id

                # Descriptor 3
                matches = bf.knnMatch(des_6, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[6]:
                    best_count[6] = count
                    best_song[6] = query_id

                # Descriptor 3
                matches = bf.knnMatch(des_7, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[7]:
                    best_count[7] = count
                    best_song[7] = query_id

                # Descriptor 8
                matches = bf.knnMatch(des_8, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[8]:
                    best_count[8] = count
                    best_song[8] = query_id

                # Descriptor 9
                matches = bf.knnMatch(des_9, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[9]:
                    best_count[9] = count
                    best_song[9] = query_id

                # Descriptor 10
                matches = bf.knnMatch(des_10, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[10]:
                    best_count[10] = count
                    best_song[10] = query_id

                # Descriptor 11
                matches = bf.knnMatch(des_11, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[11]:
                    best_count[11] = count
                    best_song[11] = query_id

                # Descriptor 12
                matches = bf.knnMatch(des_12, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[12]:
                    best_count[12] = count
                    best_song[12] = query_id

                # Descriptor 13
                matches = bf.knnMatch(des_13, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[13]:
                    best_count[13] = count
                    best_song[13] = query_id

                # Descriptor 14
                matches = bf.knnMatch(des_14, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[14]:
                    best_count[14] = count
                    best_song[14] = query_id

                # Descriptor 15
                matches = bf.knnMatch(des_15, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[15]:
                    best_count[15] = count
                    best_song[15] = query_id

                # Descriptor 16
                matches = bf.knnMatch(des_16, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[16]:
                    best_count[16] = count
                    best_song[16] = query_id

                # Descriptor 17
                matches = bf.knnMatch(des_17, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[17]:
                    best_count[17] = count
                    best_song[17] = query_id

                # Descriptor 18
                matches = bf.knnMatch(des_18, query_descriptor, k=2)
                count = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        count += 1
                if count > best_count[18]:
                    best_count[18] = count
                    best_song[18] = query_id

            for descriptor_id in range(19):
                db_util.execute_insert("INSERT INTO song_match (song_id, best_song, best_count, type) VALUES ('"+str(song_id)+"','"+str(best_song[descriptor_id])+"','"+str(best_count[descriptor_id])+"','"+str(descriptor_id)+"')")
        db_util.close()
        gc.collect()

    @staticmethod
    def match_clip(job_id):
        clip_path = "/home/rms_sys_user/rms_using_sift/test/1667.mp3"
        date_time = datetime.datetime(2019, 10, 23, 0, 0, 0)
        db_util = DatabaseUtil()
        gc.enable()
        if os.path.isfile(clip_path):
            frame_id = db_util.execute_insert(
                "INSERT INTO frame_match (job_id, start, best_song, best_count) VALUES ('" + str(
                    job_id) + "','" + date_time.strftime('%Y-%m-%d %H-%M-%S') + "','0','0')")
            audio_stream, _ = librosa.load(clip_path)
            AudioUtil.match_frame(audio_stream, frame_id)
        db_util.execute_update("UPDATE job SET status='3' WHERE id='" + str(job_id) + "'")
        db_util.close()

    @staticmethod
    def match_frame(audio_stream, frame_id):
        db_util = DatabaseUtil()
        song_list = db_util.execute_query("SELECT id,des_0 FROM song")
        clip_descriptor = AudioUtil.generate_descriptors(audio_stream)
        best_song = -1
        best_count = -1
        for (id, descriptor) in song_list:
            test_descriptor = np.frombuffer(descriptor, dtype=np.float32)
            test_descriptor = test_descriptor.reshape(-1, 128)
            bf = cv.BFMatcher()
            matches = bf.knnMatch(clip_descriptor, test_descriptor,  k=2)
            count = 0
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    count += 1

            if count > best_count:
                best_count = count
                best_song = id
        db_util.execute_update("UPDATE frame_match SET best_song='" + str(best_song) + "',best_count='" + str(
            best_count) + "' where id='" + str(frame_id) + "'")
        db_util.close()
        gc.collect()

    @staticmethod
    def register_songs(job_id):
        gc.enable()
        thread_pool = Pool(24)
        db_util = DatabaseUtil()
        number_of_songs = db_util.execute_query("SELECT COUNT(id) FROM song")
        number_of_songs = number_of_songs[0][0]
        file_directory = "/var/www/html/osca/storage/app/public/songs/"
        processes = []
        if number_of_songs == 0:
            existing_songs = DatabaseUtil.get_song_list()
            for (id, title) in existing_songs:
                thread = thread_pool.apply_async(AudioUtil.insert_song, (id, title, file_directory))
                processes.append(thread)
        for i in processes:
            i.get()
        thread_pool.close()
        thread_pool.join()
        db_util.execute_update("UPDATE job SET status='3' WHERE id='" + str(job_id) + "'")

    @staticmethod
    def insert_song(id, title, file_directory):
        db_util = DatabaseUtil()
        song_file = file_directory + str(id) + ".mp3"
        if os.path.isfile(song_file):
            audio_stream, sample_rate = librosa.load(song_file)
            descriptor = AudioUtil.generate_descriptors(audio_stream)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("INSERT into song (id, title, des_0) VALUES "
                                        + "('" + str(id) + "','" + str(title) + "',%s)", byte_string)

            audio_fast_1 = librosa.effects.time_stretch(audio_stream, 1.1)
            descriptor = AudioUtil.generate_descriptors(audio_fast_1)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_1=%s WHERE id = '"+str(id)+"'", byte_string)

            audio_fast_2 = librosa.effects.time_stretch(audio_stream, 1.2)
            descriptor = AudioUtil.generate_descriptors(audio_fast_2)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_2=%s WHERE id = '"+str(id)+"'", byte_string)
            
            audio_fast_3 = librosa.effects.time_stretch(audio_stream, 1.5)
            descriptor = AudioUtil.generate_descriptors(audio_fast_3)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_3=%s WHERE id = '"+str(id)+"'", byte_string)
            
            audio_slow_1 = librosa.effects.time_stretch(audio_stream, 0.9)
            descriptor = AudioUtil.generate_descriptors(audio_slow_1)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_4=%s WHERE id = '"+str(id)+"'", byte_string)
            
            audio_slow_2 = librosa.effects.time_stretch(audio_stream, 0.8)
            descriptor = AudioUtil.generate_descriptors(audio_slow_2)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_5=%s WHERE id = '"+str(id)+"'", byte_string)
            
            audio_slow_3 = librosa.effects.time_stretch(audio_stream, 0.5)
            descriptor = AudioUtil.generate_descriptors(audio_slow_3)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_6=%s WHERE id = '"+str(id)+"'", byte_string)

            audio_fast_1 = librosa.effects.pitch_shift(audio_stream, sample_rate, n_steps=2)
            descriptor = AudioUtil.generate_descriptors(audio_fast_1)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_7=%s WHERE id = '"+str(id)+"'", byte_string)
            
            audio_fast_2 = librosa.effects.pitch_shift(audio_stream, sample_rate, n_steps=4)
            descriptor = AudioUtil.generate_descriptors(audio_fast_2)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_8=%s WHERE id = '"+str(id)+"'", byte_string)
            
            audio_fast_3 = librosa.effects.pitch_shift(audio_stream, sample_rate, n_steps=10)
            descriptor = AudioUtil.generate_descriptors(audio_fast_3)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_9=%s WHERE id = '"+str(id)+"'", byte_string)

            audio_slow_1 = librosa.effects.pitch_shift(audio_stream, sample_rate, n_steps=-2)
            descriptor = AudioUtil.generate_descriptors(audio_slow_1)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_10=%s WHERE id = '"+str(id)+"'", byte_string)
            
            audio_slow_2 = librosa.effects.pitch_shift(audio_stream, sample_rate, n_steps=-4)
            descriptor = AudioUtil.generate_descriptors(audio_slow_2)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_11=%s WHERE id = '"+str(id)+"'", byte_string)
            
            audio_slow_3 = librosa.effects.pitch_shift(audio_stream, sample_rate, n_steps=-10)
            descriptor = AudioUtil.generate_descriptors(audio_slow_3)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_12=%s WHERE id = '"+str(id)+"'", byte_string)

            audio_fast_1 = librosa.effects.time_stretch(audio_fast_1, 1.1)
            descriptor = AudioUtil.generate_descriptors(audio_fast_1)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_13=%s WHERE id = '" + str(id) + "'", byte_string)

            audio_fast_2 = librosa.effects.time_stretch(audio_fast_2, 1.2)
            descriptor = AudioUtil.generate_descriptors(audio_fast_2)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_14=%s WHERE id = '" + str(id) + "'", byte_string)

            audio_fast_3 = librosa.effects.time_stretch(audio_fast_3, 1.5)
            descriptor = AudioUtil.generate_descriptors(audio_fast_3)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_15=%s WHERE id = '" + str(id) + "'", byte_string)

            audio_slow_1 = librosa.effects.time_stretch(audio_slow_1, 0.9)
            descriptor = AudioUtil.generate_descriptors(audio_slow_1)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_16=%s WHERE id = '" + str(id) + "'", byte_string)

            audio_slow_2 = librosa.effects.time_stretch(audio_slow_2, 0.8)
            descriptor = AudioUtil.generate_descriptors(audio_slow_2)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_17=%s WHERE id = '" + str(id) + "'", byte_string)

            audio_slow_3 = librosa.effects.time_stretch(audio_slow_3, 0.5)
            descriptor = AudioUtil.generate_descriptors(audio_slow_3)
            byte_string = descriptor.tobytes()
            db_util.execute_insert_blob("UPDATE song SET des_18=%s WHERE id = '" + str(id) + "'", byte_string)


        db_util.close()
        gc.collect()

    @staticmethod
    def generate_descriptors(audio_stream):
        # Generate STFT spectrogram
        stft_spectrogram = np.abs(librosa.stft(audio_stream, win_length=2048, hop_length=1024))
        librosa.display.specshow(librosa.amplitude_to_db(stft_spectrogram, ref=np.max), y_axis='off', x_axis='off')
        plt.tight_layout()

        # Save spectrogram to File Buffer
        file_variable = io.BytesIO()
        plt.savefig(file_variable, bbox_inches='tight')
        plt.close()
        file_variable.seek(0)
        file_bytes = np.asarray(bytearray(file_variable.read()), dtype=np.uint8)

        # Read File Buffer as OpenCV Image
        output_image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

        # Generate Descriptors
        sift = cv.xfeatures2d.SIFT_create()
        _, descriptors = sift.detectAndCompute(output_image, None)
        gc.collect()

        # Return Descriptors
        return descriptors
