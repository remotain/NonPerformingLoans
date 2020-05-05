# Data challenge: Non Performing Loans

This Data Challenge was part of the interview process for the position of Senior Data Scientist. The challenge was meant to demonstrate the required technical skills for the position. Thanks to the work performed to solve this Challenge, **I manage to obtain with a job offer**.

This public repository contain all the work done toward this Data Challenge and served to guide the interviewers through the general strategy, the methodology as well as the code.

I prefer to not disclose the name of the company and the original dataset provided.

The results of this work is summarised in two main documents:
1. [**Business summary**](Summary_Slides.pdf)
2. [**Technical summary**](Summary_Report.pdf)

## Executive summary

Two supervised machine learning algorithms, trained on the provided historical dataset, are employed to estimate the likelihood to repay and the recovery rate in a 12 months ahead window. The model predictions are then used to re-rank NPL customers. Given the NPL sample as of 09-2017, the expected 12 months recovery rate is between 1.6M€ and 6M€.

## Framework

A non-performing loan (NPL) is the sum of borrowed money upon which the debtor has not made his scheduled payments for at least 90 days. Once a loan is non-performing, the odds that it will be fully repaid are considered to be substantially lower. High levels of NPLs inhibit the capacity of banks to lend to the economy and take up valuable bank management time. As of the third quarter of 2016, NPLs of significant institutions in the Euro area amounted to €921bln (average NPE of 7%), therefore the ECB asks banks to devise a strategy to manage and reduce the volume of impaired loans.

## Objective

The general goal of this data challenge is to help NPL portfolios managers in their daily tasks by re-ranking their customers according to the likelihood to repay and/or by suggesting new strategies to maximize the recovery rate. In particular, you are asked to perform analysis on the given data, train predictive models and present your findings in a clear and data-driven way.