FROM python

ARG year
ARG bucket_name
ARG ACCESS_KEY_ID
ARG ACCESS_SECRET_KEY

RUN pip install pandas
RUN pip install boto3
COPY part2script.py .
RUN python3 part2script.py $year $bucket_name $ACCESS_KEY_ID $ACCESS_SECRET_KEY
