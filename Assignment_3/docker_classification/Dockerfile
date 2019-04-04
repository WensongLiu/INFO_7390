FROM python:3

ARG year
ARG quater
ARG bucket_name
ARG ACCESS_KEY_ID
ARG ACCESS_SECRET_KEY
ARG username
ARG password
ARG model


RUN pip install pandas
RUN pip install boto3
RUN pip install sklearn
RUN pip install requests
RUN pip install tabulate
RUN pip install "colorama>=0.3.8"
RUN pip install future
RUN pip uninstall h2o
RUN pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
RUN pip install deap update_checker tqdm stopit
RUN pip install xgboost
RUN pip install tpot


COPY login_download_unzip.py .
COPY roc_confusion.py .
COPY train_save_model.py .

CMD python3 login_download_unzip.py $year $quater $username $password
CMD python3 train_save_model.py $year $quater $bucket_name $ACCESS_KEY_ID $ACCESS_SECRET_KEY $model
CMD python3 roc_confusion.py $year $quater