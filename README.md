# Machine Learning Pipeline for Mortgage Backed Securities Prepayment Risk

## Description
This project aims to predict loan pre-payment likelihood based on various factors such as Equated Monthly Installment (EMI), total payment, interest amount, monthly income, current principal, and whether the person has ever been delinquent in their loan payments. The goal is to provide insights into the likelihood of pre-payment and assist in making informed decisions related to loan repayment strategies.

## Dataset
The data dictionary provides an overview of the dataset used in this analysis. It describes the 28 columns present in the dataset, including details such as column names, data types, and a brief explanation of each column's meaning. Understanding the data dictionary is crucial for interpreting and analyzing the dataset accurately.

The dataset used for this analysis contains information about borrowers and their loan details, including EMI, total payment, interest amount, monthly income, current principal, and whether they have been delinquent in their loan payments. The dataset is in a tabular format and is provided as a CSV file called LoanExport.csv.
The data is obtained from Freddie Mac's official portal for home loans. The size of the home loan data is (291452 x 28). It contains 291452 data points and 28 columns or parameters denoting different data features. Some of the noteworthy features of the dataset are:

0 CreditScore : Credit score of the client

1 FirstPaymentDate : First payment date of the customer

2 FirstTimeHomebuyer : If the customer is first time home buyer

3 MaturityDate : Maturity date of the customer

4 MSA : Mortgage security amount

5 MIP : Mortgage insurance percentage

6 Units : Number of units

7 Occupancy : Occupancy status at the time the loan

8 OCLTV : Original Combined Loan-to-Value

9 DTI : Debt to income ratio

10 OrigUPB : Original unpaid principal balance

11 LTV : Loan-to-Value

12 OrigInterestRate : Original interest rate

13 Channel : The origination channel used by the party

14 PPM : Prepayment penalty mortgage

15 ProductType : Type of product

16 PropertyState : State in which the property is located

17 PropertyType : Property type

18 PostalCode : Postal code of the property

19 LoanSeqNum : Loan number

20 LoanPurpose : Purpose of the loan

21 OrigLoanTerm : Original term of the loan

22 NumBorrowers : Number of borrowers

23 SellerName : Name of seller

24 ServicerName : Name of the service used

25 EverDelinquent : If the loan was ever delinquent

26 MonthsDelinquent : Months of delinquent

27 MonthsInRepayment : Months in repayment
