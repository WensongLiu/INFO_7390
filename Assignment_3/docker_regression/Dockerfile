FROM python:3

ARG year
ARG quater
ARG username
ARG password


RUN pip install pandas
RUN pip install boto3
RUN pip install sklearn
RUN pip install requests-html
RUN pip install mlxtend
COPY part_2_pipeline.py .
CMD python3 classification.py $year $quater $username $password
