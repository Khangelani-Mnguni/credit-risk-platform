SCHEMA = {

    # =====================================================
    # IDENTIFIERS
    # =====================================================
    "unnamed:_0": "float64",
    "id": "string",

    # =====================================================
    # LOAN DETAILS
    # =====================================================
    "loan_amnt": "float64",
    "funded_amnt": "float64",
    "funded_amnt_inv": "float64",
    "term": "string",
    "int_rate": "string",
    "installment": "float64",
    "grade": "string",
    "sub_grade": "string",
    "loan_status": "string",
    "purpose": "string",
    "title": "string",
    "pymnt_plan": "string",
    "url": "string",

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
    # DATES
    # =====================================================
    "issue_d": "string",
    "earliest_cr_line": "string",
    "last_pymnt_d": "string",
    "next_pymnt_d": "string",
    "last_credit_pull_d": "string",

    # =====================================================
    # CREDIT HISTORY
    # =====================================================
    "fico_range_low": "float64",
    "fico_range_high": "float64",
    "last_fico_range_low": "float64",
    "last_fico_range_high": "float64",

    # =====================================================
    # DELINQUENCIES & DEROGATORY EVENTS
    # =====================================================
    "delinq_2yrs": "float64",
    "mths_since_last_delinq": "float64",
    "mths_since_last_record": "float64",
    "mths_since_last_major_derog": "float64",
    "acc_now_delinq": "float64",
    "delinq_amnt": "float64",

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

    # =====================================================
    # REVOLVING CREDIT
    # =====================================================
    "revol_bal": "float64",
    "revol_util": "string",
    "total_rev_hi_lim": "float64",

    # =====================================================
    # PAYMENTS & RECOVERIES
    # =====================================================
    "out_prncp": "float64",
    "out_prncp_inv": "float64",
    "total_pymnt": "float64",
    "total_pymnt_inv": "float64",
    "total_rec_prncp": "float64",
    "total_rec_int": "float64",
    "total_rec_late_fee": "float64",
    "recoveries": "float64",
    "collection_recovery_fee": "float64",
    "last_pymnt_amnt": "float64",

    # =====================================================
    # POLICY & APPLICATION
    # =====================================================
    "policy_code": "float64",
    "application_type": "string",
    "initial_list_status": "string",

    # =====================================================
    # JOINT APPLICATIONS
    # =====================================================
    "annual_inc_joint": "float64",
    "dti_joint": "float64",
    "verification_status_joint": "string",

    # =====================================================
    # BALANCES & EXPOSURES
    # =====================================================
    "tot_coll_amt": "float64",
    "tot_cur_bal": "float64",
    "avg_cur_bal": "float64",
    "tot_hi_cred_lim": "float64",
    "total_bal_ex_mort": "float64",
    "total_bc_limit": "float64",
    "total_il_high_credit_limit": "float64",

    # =====================================================
    # INSTALLMENT LOANS
    # =====================================================
    "open_acc_6m": "float64",
    "open_act_il": "float64",
    "open_il_12m": "float64",
    "open_il_24m": "float64",
    "mths_since_rcnt_il": "float64",
    "total_bal_il": "float64",
    "il_util": "float64",

    # =====================================================
    # REVOLVING TRADELINES
    # =====================================================
    "open_rv_12m": "float64",
    "open_rv_24m": "float64",
    "max_bal_bc": "float64",
    "all_util": "float64",

    # =====================================================
    # UTILIZATION & BEHAVIOR
    # =====================================================
    "bc_open_to_buy": "float64",
    "bc_util": "float64",
    "pct_tl_nvr_dlq": "float64",
    "percent_bc_gt_75": "float64",

    # =====================================================
    # MORTGAGE
    # =====================================================
    "mort_acc": "float64",

    # =====================================================
    # ACCOUNT AGE METRICS
    # =====================================================
    "mo_sin_old_il_acct": "float64",
    "mo_sin_old_rev_tl_op": "float64",
    "mo_sin_rcnt_rev_tl_op": "float64",
    "mo_sin_rcnt_tl": "float64",

    # =====================================================
    # RECENCY METRICS
    # =====================================================
    "mths_since_recent_bc": "float64",
    "mths_since_recent_bc_dlq": "float64",
    "mths_since_recent_inq": "float64",
    "mths_since_recent_revol_delinq": "float64",

    # =====================================================
    # TRADELINE COUNTS
    # =====================================================
    "num_accts_ever_120_pd": "float64",
    "num_actv_bc_tl": "float64",
    "num_actv_rev_tl": "float64",
    "num_bc_sats": "float64",
    "num_bc_tl": "float64",
    "num_il_tl": "float64",
    "num_op_rev_tl": "float64",
    "num_rev_accts": "float64",
    "num_rev_tl_bal_gt_0": "float64",
    "num_sats": "float64",
    "num_tl_120dpd_2m": "float64",
    "num_tl_30dpd": "float64",
    "num_tl_90g_dpd_24m": "float64",
    "num_tl_op_past_12m": "float64",

    # =====================================================
    # PUBLIC RECORDS
    # =====================================================
    "pub_rec_bankruptcies": "float64",
    "tax_liens": "float64",

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
    "sec_app_open_act_il": "float64",
    "sec_app_num_rev_accts": "float64",
    "sec_app_chargeoff_within_12_mths": "float64",
    "sec_app_collections_12_mths_ex_med": "float64",

    # =====================================================
    # HARDSHIP PROGRAM
    # =====================================================
    "hardship_flag": "string",
    "hardship_type": "string",
    "hardship_reason": "string",
    "hardship_status": "string",
    "deferral_term": "float64",
    "hardship_amount": "float64",
    "hardship_start_date": "string",
    "hardship_end_date": "string",
    "payment_plan_start_date": "string",
    "hardship_length": "float64",
    "hardship_dpd": "float64",
    "hardship_loan_status": "string",
    "orig_projected_additional_accrued_interest": "float64",
    "hardship_payoff_balance_amount": "float64",
    "hardship_last_payment_amount": "float64",

    # =====================================================
    # DEBT SETTLEMENT
    # =====================================================
    "debt_settlement_flag": "string"
}