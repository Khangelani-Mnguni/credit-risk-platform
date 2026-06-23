SCHEMA = {

    # =====================================================
    # IDENTIFIERS
    # =====================================================
    "id": "string",
    "member_id": "float64",

    # =====================================================
    # LOAN DETAILS
    # =====================================================
    "loan_amnt": "float64",
    "funded_amnt": "float64",
    "funded_amnt_inv": "float64",
    "term": "string",
    "int_rate": "float64",
    "installment": "float64",
    "grade": "string",
    "sub_grade": "string",
    "loan_status": "string",
    "purpose": "string",
    "title": "string",

    # =====================================================
    # EMPLOYMENT & INCOME
    # =====================================================
    "emp_title": "string",
    "emp_length": "string",
    "annual_inc": "float64",
    "verification_status": "string",

    # =====================================================
    # BORROWER PROFILE
    # =====================================================
    "home_ownership": "string",
    "zip_code": "string",
    "addr_state": "string",
    "dti": "float64",

    # =====================================================
    # CREDIT HISTORY
    # =====================================================
    "earliest_cr_line": "string",
    "fico_range_low": "float64",
    "fico_range_high": "float64",
    "last_fico_range_low": "float64",
    "last_fico_range_high": "float64",

    # =====================================================
    # DELINQUENCY & DEFAULT HISTORY
    # =====================================================
    "delinq_2yrs": "float64",
    "mths_since_last_delinq": "float64",
    "mths_since_last_record": "float64",
    "mths_since_last_major_derog": "float64",

    # =====================================================
    # INQUIRIES
    # =====================================================
    "inq_last_6mths": "float64",
    "inq_fi": "float64",
    "inq_last_12m": "float64",

    # =====================================================
    # ACCOUNT COUNTS
    # =====================================================
    "open_acc": "float64",
    "pub_rec": "float64",
    "total_acc": "float64",
    "acc_now_delinq": "float64",

    # =====================================================
    # REVOLVING CREDIT
    # =====================================================
    "revol_bal": "float64",
    "revol_util": "float64",
    "total_rev_hi_lim": "float64",

    # =====================================================
    # PAYMENT & PERFORMANCE
    # =====================================================
    "out_prncp": "float64",
    "out_prncp_inv": "float64",
    "total_pymnt": "float64",
    "total_pymnt_inv": "float64",
    "total_rec_prncp": "float64",
    "total_rec_int": "float64",
    "total_rec_late_fee": "float64",
    "last_pymnt_amnt": "float64",

    # =====================================================
    # RECOVERIES
    # =====================================================
    "recoveries": "float64",
    "collection_recovery_fee": "float64",

    # =====================================================
    # IMPORTANT DATES
    # =====================================================
    "issue_d": "string",
    "last_pymnt_d": "string",
    "next_pymnt_d": "string",
    "last_credit_pull_d": "string",

    # =====================================================
    # JOINT APPLICATIONS
    # =====================================================
    "application_type": "string",
    "annual_inc_joint": "float64",
    "dti_joint": "float64",
    "verification_status_joint": "string",

    # =====================================================
    # BALANCES & EXPOSURES
    # =====================================================
    "tot_coll_amt": "float64",
    "tot_cur_bal": "float64",
    "tot_hi_cred_lim": "float64",
    "total_bal_ex_mort": "float64",
    "total_bc_limit": "float64",
    "total_il_high_credit_limit": "float64",

    # =====================================================
    # MORTGAGE & PROPERTY
    # =====================================================
    "mort_acc": "float64",

    # =====================================================
    # CREDIT UTILIZATION METRICS
    # =====================================================
    "avg_cur_bal": "float64",
    "bc_open_to_buy": "float64",
    "bc_util": "float64",
    "all_util": "float64",

    # =====================================================
    # SECONDARY APPLICANT
    # =====================================================
    "revol_bal_joint": "float64",
    "sec_app_fico_range_low": "float64",
    "sec_app_fico_range_high": "float64",
    "sec_app_earliest_cr_line": "string",
    "sec_app_inq_last_6mths": "float64",
    "sec_app_mort_acc": "float64",
    "sec_app_open_acc": "float64",
    "sec_app_revol_util": "float64",

    # =====================================================
    # HARDSHIP PROGRAM
    # =====================================================
    "hardship_flag": "string",
    "hardship_type": "string",
    "hardship_reason": "string",
    "hardship_status": "string",
    "hardship_start_date": "string",
    "hardship_end_date": "string",
    "payment_plan_start_date": "string",
    "hardship_amount": "float64",
    "hardship_length": "float64",
    "hardship_dpd": "float64",
    "hardship_loan_status": "string",

    # =====================================================
    # DEBT SETTLEMENT
    # =====================================================
    "debt_settlement_flag": "string",
    "debt_settlement_flag_date": "string",
    "settlement_status": "string",
    "settlement_date": "string",
    "settlement_amount": "float64",
    "settlement_percentage": "float64",
    "settlement_term": "float64"
}