''' install pandas : pip install pandas '''
import pandas as pd

''' install flask : pip install Flask '''
from flask import Flask, Response, render_template, jsonify
import time
import json

import numpy as np


''' pip install hmmlearn '''
from hmmlearn import hmm

''' pip install python_speech_features '''
from python_speech_features import mfcc

''' pip install matplotlib '''
# import matplotlib.pyplot as plt

from scipy.io import wavfile
from HMMTrainer import HMMTrainer

import mysql.connector
host = "localhost"
user = "root"
password = ""
db_name = "skripsi_yanuar"


app = Flask(__name__)

chosen = ["angry", "sad", "neutral", "fear", "happy"]

def processing():
    ''' Read model .csv '''
    data = pd.read_csv("data-sekunder.csv")

    # drop column Unnamed: 0
    data = data.drop('Unnamed: 0', axis=1)

    # get labels and features
    X = data.drop('Emotions', axis=1).values
    y = data['Emotions'].values

    print(f"Label terkini {np.unique(y)}")

    print("Start Sinkronisasi Model - HMM")
    # start processing
    hmm_models = []
    nilai_ekstraksi = []

    i = 0
    for data_x in X:

        label = y[i]
        detailFrame = []
        for perFrame in data_x:

            array_detail_perFrame = []
            exp = str(perFrame).split(", ")

            if (len(exp) > 1):
                for isi in exp:
                    if (isi != "nan"):
                        # array_detail_perFrame = np.append(array_detail_perFrame, float(isi))
                        array_detail_perFrame.append(float(isi))

            if len(array_detail_perFrame) > 0:
                # detailFrame = np.append(detailFrame, array_detail_perFrame, axis=0)
                detailFrame.append(array_detail_perFrame)

        hasil_mfcc = np.array(detailFrame)

        if len(nilai_ekstraksi) == 0:
            nilai_ekstraksi = np.array(hasil_mfcc)

        else:

            nilai_ekstraksi = np.append(nilai_ekstraksi, hasil_mfcc, axis=0)

            # Train and save HMM model
            hmm_trainer = HMMTrainer()
            hmm_trainer.train(nilai_ekstraksi)
            # hmm_models["hmm_trainer"] = hmm_trainer

            # print(type(hmm_trainer))

            hmm_models.append([hmm_trainer, label])

        print(f"Proses ke-{i} = {len(nilai_ekstraksi)} fitur berhasil diolah")
        i += 1


    return hmm_models

def ekstraksi_mfcc(file):
    sampling_freq, audio = wavfile.read(file)

    # ekstraksi mfcc
    mfcc_features = mfcc(audio, sampling_freq, nfft=1200)
    return mfcc_features


def proses_klasifikasi(input):

    start_time = time.time()
    mfcc_features = ekstraksi_mfcc(input)

    max_score = None
    output_label = None

    klasifikasi = [0, 0, 0, 0, 0]

    hmm_models = processing()

    for item in hmm_models:
        hmm_model, label = item
        score = hmm_model.get_score(mfcc_features)
        # print(score)

        if max_score is None or score > max_score:
            max_score = score
            output_label = label

            keyIndex = chosen.index(label)

            klasifikasi[keyIndex] += 1

    total = sum(klasifikasi)

    hasil_label = []
    hasil_persentase = []

    for key in range(0, len(chosen)):
        label = chosen[key]
        hasil = klasifikasi[key] / total * 100

        hasil_label.append(label)
        hasil_persentase.append(hasil)

    # time execution
    end_time = time.time()

    # insert data
    # df_label = pd.DataFrame(hasil_label)
    # df_persentase = pd.DataFrame(hasil_persentase)
    try:
        connection = mysql.connector.connect(host=host,
                                             database=db_name,
                                             user=user,
                                             password=password)
        cursor = connection.cursor()

        # - - - - - - - - - - - -
        # update status
        time_execute = end_time - start_time

        dt_hasilkeseluruhan = {

            'label' : hasil_label ,
            'persentase' : hasil_persentase
        }

        dt_json = json.dumps(dt_hasilkeseluruhan, indent=4)
        # print(dt_json)

        # Update single record now
        sql_update_query = f"""Update klasifikasi SET time_execution={time_execute}, data_json='{dt_json}',status=1 WHERE id=1"""
        cursor.execute(sql_update_query)
        connection.commit()


    except mysql.connector.Error as error:
        print("Failed to UPDATE record into MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

            print("MySQL connection is closed")


    return "Ok"


@app.route("/eksekusi/<string:filename>")
def eksekusi( filename ):

    input = f"C:\\xampp\\htdocs\\yanuar-voice-reconigtion\\assets\\wav\\{filename}"
    # input = f"C:/xampp-v8/htdocs/sinauka-research-speechrecognition/assets/wav/{filename}"

    # input = "YAF_back_sad.wav"
    result = proses_klasifikasi(input)
    return jsonify(result)



@app.route("/pengujian/<int:training>/<int:testing>")
def pengujian(training, testing):
    ''' CM '''

    # open model
    awal = time.time()

    ''' read data csv '''
    data = pd.read_csv("data-sekunder1.csv")
    data = data.drop("Unnamed: 0", axis=1)

    X = data.drop("Emotions", axis=1).values
    y = data["Emotions"].values


    # split data
    data_terkini = len(X)
    jml_training = int((training / 100) * data_terkini)
    jml_testing = int((testing / 100) * data_terkini)

    dt_training = X[:jml_training]
    dt_testing = X[jml_training:]


    # hmm
    print("Sedang memproses HMM")
    hmm_models = []
    nilai_ekstraksi = []

    i = 0
    for data_x in dt_training:

        print(f"HMM training ke {i}")
        label = y[i]
        detailFrame = []
        for perFrame in data_x:

            array_detail_perFrame = []
            exp = str(perFrame).split(", ")

            if (len(exp) > 1):
                for isi in exp:
                    if ( isi != "nan" ):
                        array_detail_perFrame.append(float(isi))


            if ( len(array_detail_perFrame) > 0 ):
                detailFrame.append(array_detail_perFrame)

        hasil_mfcc = np.array(detailFrame)
        if ( len(nilai_ekstraksi) == 0 ):
            nilai_ekstraksi = np.array(hasil_mfcc)

        else:

            nilai_ekstraksi = np.append(nilai_ekstraksi, hasil_mfcc, axis=0)

            # Train and save HMM model
            hmm_trainer = HMMTrainer()
            hmm_trainer.train(nilai_ekstraksi)

            hmm_models.append([hmm_trainer, label])

        i += 1


    print("HMM selesai")
    print("Proses Klasifikasi")

    #klasifikasi

    label_aktual = []
    label_prediksi = []

    for data_x in dt_testing:

        ds_label = y[i]
        detailFrame = []
        for perFrame in data_x:

            array_detail_perFrame = []
            exp = str(perFrame).split(", ")

            if (len(exp) > 1):
                for isi in exp:
                    if (isi != "nan"):
                        array_detail_perFrame.append(float(isi))

            if (len(array_detail_perFrame) > 0):
                detailFrame.append(array_detail_perFrame)

        hasil_mfcc = np.array(detailFrame)

        # klasifikasi per data

        max_score = None
        output_label = None
        klasifikasi = [0, 0, 0, 0, 0]

        for item in hmm_models:

            hmm_model, label = item
            score = hmm_model.get_score(hasil_mfcc)

            if max_score is None or score > max_score:
                max_score = score
                output_label = label

                keyIndex = chosen.index(label)

                klasifikasi[keyIndex] += 1
        total = sum(klasifikasi)
        hasil_label = []

        hasil_persentase = []

        for key in range(0, len(chosen)):
            label = chosen[key]
            hasil = klasifikasi[key] / total * 100

            hasil_label.append(label)
            hasil_persentase.append(hasil)


        # get label
        max_value  = max(hasil_persentase)
        getIndex   = hasil_persentase.index(max_value)
        finalLabel = chosen[getIndex]


        label_aktual.append(ds_label)
        label_prediksi.append(finalLabel)
        print(f"Pengujian data ke-{i} dengan label {ds_label} terprediksi {finalLabel}")

        i += 1


    # pembentukan confusion matrix

    # inisialisasi matrix dengan nilai 0
    init_baris = []
    num_baris = 0
    for baris in chosen:

        # pembuatan kolom
        num_kolom = 0
        init_kolom = []
        for kolom in chosen:

            init_kolom.append(0)

            num_kolom += 1


        init_baris.append(init_kolom)
        num_baris += 1

    urutan = 0
    for actual in label_aktual:

        predict = label_prediksi[urutan]

        # get index
        index_actual = chosen.index(actual)
        index_predict = chosen.index(predict)

        init_baris[index_actual][index_predict] += 1

        urutan += 1


    # print(init_baris)
    array_to_string = ""
    for baris in init_baris:
        string_ints = [str(int) for int in baris]
        isi_string = " ".join(string_ints)

        array_to_string += (isi_string + ",")

    print(array_to_string)
    print(type(array_to_string))

    try:
        connection = mysql.connector.connect(host=host,
                                             database=db_name,
                                             user=user,
                                             password=password)
        cursor = connection.cursor()

        # - - - - - - - - - - - -
        # update status

        berakhir = time.time()
        time_execute = berakhir - awal


        # Update single record now
        sql_update_query = f"""Update pengujian SET waktu={time_execute}, json='{array_to_string}', training='{training}', testing='{testing}' WHERE id='1'"""
        cursor.execute(sql_update_query)
        connection.commit()


    except mysql.connector.Error as error:
        print("Failed to UPDATE record into MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

            print("MySQL connection is closed")





    berakhir = time.time()
    return f"ok eksekusi time {berakhir - awal}"



if __name__ == '__main__':
    app.run(debug=True)