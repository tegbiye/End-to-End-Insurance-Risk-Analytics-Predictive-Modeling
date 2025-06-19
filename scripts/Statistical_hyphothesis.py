import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from load_data import load_csv


def statistic_hyphotesis_test(df, alpha):
    try:
        # Convert numeric columns to appropriate types, coercing errors
        df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
        df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')

        # Drop rows where essential numeric data is missing
        df.dropna(subset=['TotalPremium', 'TotalClaims', 'Province',
                          'PostalCode', 'Gender'], inplace=True)

        # Handle empty strings in 'Gender' and 'Citizenship' by treating them as NaN and then dropping or imputing
        # For 'Gender', you might want to map 'Mr'/'Mrs'/'Ms' from 'Title' column if Gender is mostly empty.
        # For this script, we'll just drop empty/whitespace strings.
        df['Gender'].replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        # Drop rows where gender is still missing/empty
        df.dropna(subset=['Gender'], inplace=True)

        # Create 'HasClaim' indicator: 1 if TotalClaims > 0, else 0
        df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)

        # Calculate Margin: TotalPremium - TotalClaims
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']

        # Filter for Claim Severity: only consider policies with claims
        df_claims_only = df[df['HasClaim'] == 1].copy()

        # --- Hypothesis Tests ---

        # Hypothesis 1: H₀: There are no risk differences across provinces (Claim Frequency & Claim Severity)
        print("--- Hypothesis 1: Risk Differences across Provinces ---")

        # 1a. Claim Frequency by Province (Chi-squared test)
        if not df['Province'].empty:
            province_claim_freq_table = pd.crosstab(
                df['Province'], df['HasClaim'])
            if province_claim_freq_table.shape[0] > 1 and province_claim_freq_table.shape[1] > 1:
                chi2, p_val_freq_province, _, _ = stats.chi2_contingency(
                    province_claim_freq_table)
                print(
                    f"Claim Frequency by Province (Chi-squared test): p-value = {p_val_freq_province:.4f}")
                if p_val_freq_province < alpha:
                    print(
                        f"  Reject H₀: There is a significant difference in claim frequency across provinces (p < {alpha}).")
                else:
                    print(
                        f"  Fail to reject H₀: No significant difference in claim frequency across provinces (p >= {alpha}).")
            else:
                print(
                    "  Not enough variation in 'Province' or 'HasClaim' to perform Chi-squared test for claim frequency.")
        else:
            print("  'Province' column is empty after cleaning.")

        # 1b. Claim Severity by Province (ANOVA)
        if not df_claims_only['Province'].empty:
            # Check if there's enough data and variance within provinces for ANOVA
            if len(df_claims_only['Province'].unique()) > 1 and len(df_claims_only) > len(df_claims_only['Province'].unique()):
                model_severity_province = ols(
                    'TotalClaims ~ C(Province)', data=df_claims_only).fit()
                anova_table_severity_province = sm.stats.anova_lm(
                    model_severity_province, typ=2)
                p_val_severity_province = anova_table_severity_province['PR(>F)'][0]
                print(
                    f"Claim Severity by Province (ANOVA): p-value = {p_val_severity_province:.4f}")
                if p_val_severity_province < alpha:
                    print(
                        f"  Reject H₀: There is a significant difference in claim severity across provinces (p < {alpha}).")
                    # Perform Tukey's HSD post-hoc test if ANOVA is significant
                    tukey_severity_province = pairwise_tukeyhsd(endog=df_claims_only['TotalClaims'],
                                                                groups=df_claims_only['Province'],
                                                                alpha=alpha)
                    print("  Tukey's HSD Post-hoc Test for Claim Severity by Province:")
                    print(tukey_severity_province)
                else:
                    print(
                        f"  Fail to reject H₀: No significant difference in claim severity across provinces (p >= {alpha}).")
            else:
                print(
                    "  Not enough variation in 'Province' or data points with claims to perform ANOVA for claim severity.")
        else:
            print(
                "  No claims data available for 'Province' after cleaning.")
        print("\n" + "-"*60 + "\n")

        # Hypothesis 2: H₀: There are no risk differences between zip codes (Claim Frequency & Claim Severity)
        print("--- Hypothesis 2: Risk Differences between Zip Codes ---")

        # 2a. Claim Frequency by Zip Code (Chi-squared test)
        if not df['PostalCode'].empty:
            zip_claim_freq_table = pd.crosstab(
                df['PostalCode'], df['HasClaim'])
            if zip_claim_freq_table.shape[0] > 1 and zip_claim_freq_table.shape[1] > 1:
                # Filter out postal codes with very few observations for chi-squared stability
                min_obs = 10  # Adjust as needed
                valid_postal_codes = zip_claim_freq_table[(
                    zip_claim_freq_table > 0).sum(axis=1) > 0].index
                filtered_zip_claim_freq_table = zip_claim_freq_table.loc[valid_postal_codes]

                if filtered_zip_claim_freq_table.shape[0] > 1:
                    chi2, p_val_freq_zip, _, _ = stats.chi2_contingency(
                        filtered_zip_claim_freq_table)
                    print(
                        f"Claim Frequency by Zip Code (Chi-squared test): p-value = {p_val_freq_zip:.4f}")
                    if p_val_freq_zip < alpha:
                        print(
                            f"  Reject H₀: There is a significant difference in claim frequency between zip codes (p < {alpha}).")
                    else:
                        print(
                            f"  Fail to reject H₀: No significant difference in claim frequency between zip codes (p >= {alpha}).")
                else:
                    print("  Not enough distinct postal codes with observations to perform Chi-squared test for claim frequency after filtering sparse groups.")
            else:
                print(
                    "  Not enough variation in 'PostalCode' or 'HasClaim' to perform Chi-squared test for claim frequency.")
        else:
            print("  'PostalCode' column is empty after cleaning.")

        # 2b. Claim Severity by Zip Code (ANOVA)
        if not df_claims_only['PostalCode'].empty:
            if len(df_claims_only['PostalCode'].unique()) > 1 and len(df_claims_only) > len(df_claims_only['PostalCode'].unique()):
                # Filter out postal codes with very few observations for ANOVA stability
                # Need to ensure at least 2 data points per group for ANOVA to work
                postal_code_counts = df_claims_only['PostalCode'].value_counts(
                )
                valid_postal_codes_for_anova = postal_code_counts[postal_code_counts >= 2].index
                df_claims_only_filtered_zip = df_claims_only[df_claims_only['PostalCode'].isin(
                    valid_postal_codes_for_anova)].copy()

                if len(df_claims_only_filtered_zip['PostalCode'].unique()) > 1:
                    model_severity_zip = ols(
                        'TotalClaims ~ C(PostalCode)', data=df_claims_only_filtered_zip).fit()
                    anova_table_severity_zip = sm.stats.anova_lm(
                        model_severity_zip, typ=2)
                    p_val_severity_zip = anova_table_severity_zip['PR(>F)'][0]
                    print(
                        f"Claim Severity by Zip Code (ANOVA): p-value = {p_val_severity_zip:.4f}")
                    if p_val_severity_zip < alpha:
                        print(
                            f"  Reject H₀: There is a significant difference in claim severity between zip codes (p < {alpha}).")
                        # Perform Tukey's HSD post-hoc test if ANOVA is significant
                        tukey_severity_zip = pairwise_tukeyhsd(endog=df_claims_only_filtered_zip['TotalClaims'],
                                                               groups=df_claims_only_filtered_zip['PostalCode'], alpha=alpha)
                        print(
                            "Tukey's HSD Post-hoc Test for Claim Severity by Zip Code:")
                        print(tukey_severity_zip)
                    else:
                        print(
                            f"  Fail to reject H₀: No significant difference in claim severity between zip codes (p >= {alpha}).")
                else:
                    print(
                        "  Not enough distinct postal codes with enough claims to perform ANOVA for claim severity after filtering sparse groups.")
            else:
                print(
                    "  Not enough variation in 'PostalCode' or data points with claims to perform ANOVA for claim severity.")
        else:
            print(
                "  No claims data available for 'PostalCode' after cleaning.")
        print("\n" + "-"*60 + "\n")

        # Hypothesis 3: H₀: There are no significant margin (profit) difference between zip codes
        print("--- Hypothesis 3: Margin Difference between Zip Codes ---")
        if not df['PostalCode'].empty:
            if len(df['PostalCode'].unique()) > 1 and len(df) > len(df['PostalCode'].unique()):
                # Filter out postal codes with very few observations for ANOVA stability
                postal_code_counts_margin = df['PostalCode'].value_counts(
                )
                valid_postal_codes_for_anova_margin = postal_code_counts_margin[
                    postal_code_counts_margin >= 2].index
                df_filtered_zip_margin = df[df['PostalCode'].isin(
                    valid_postal_codes_for_anova_margin)].copy()

                if len(df_filtered_zip_margin['PostalCode'].unique()) > 1:
                    model_margin_zip = ols(
                        'Margin ~ C(PostalCode)', data=df_filtered_zip_margin).fit()
                    anova_table_margin_zip = sm.stats.anova_lm(
                        model_margin_zip, typ=2)
                    p_val_margin_zip = anova_table_margin_zip['PR(>F)'][0]
                    print(
                        f"Margin by Zip Code (ANOVA): p-value = {p_val_margin_zip:.4f}")
                    if p_val_margin_zip < alpha:
                        print(
                            f"  Reject H₀: There is a significant difference in margin between zip codes (p < {alpha}).")
                        # Perform Tukey's HSD post-hoc test if ANOVA is significant
                        tukey_margin_zip = pairwise_tukeyhsd(endog=df_filtered_zip_margin['Margin'],
                                                             groups=df_filtered_zip_margin['PostalCode'],
                                                             alpha=alpha)
                        print("  Tukey's HSD Post-hoc Test for Margin by Zip Code:")
                        print(tukey_margin_zip)
                    else:
                        print(
                            f"  Fail to reject H₀: No significant difference in margin between zip codes (p >= {alpha}).")
                else:
                    print(
                        "  Not enough distinct postal codes with enough data to perform ANOVA for margin after filtering sparse groups.")
            else:
                print(
                    "  Not enough variation in 'PostalCode' or data points to perform ANOVA for margin.")
        else:
            print("  'PostalCode' column is empty after cleaning.")
        print("\n" + "-"*60 + "\n")

        # Hypothesis 4: H₀: There are not significant risk difference between Women and Men
        print("--- Hypothesis 4: Risk Differences between Women and Men ---")

        # 4a. Claim Frequency by Gender (Chi-squared test)
        if 'Gender' in df.columns and not df['Gender'].empty:
            # Filter for valid gender values (e.g., 'Male', 'Female')
            gender_data = df[df['Gender'].isin(['Male', 'Female'])].copy()
            if not gender_data.empty and len(gender_data['Gender'].unique()) > 1:
                gender_claim_freq_table = pd.crosstab(
                    gender_data['Gender'], gender_data['HasClaim'])
                if gender_claim_freq_table.shape[0] > 1 and gender_claim_freq_table.shape[1] > 1:
                    chi2, p_val_freq_gender, _, _ = stats.chi2_contingency(
                        gender_claim_freq_table)
                    print(
                        f"Claim Frequency by Gender (Chi-squared test): p-value = {p_val_freq_gender:.4f}")
                    if p_val_freq_gender < alpha:
                        print(
                            f"  Reject H₀: There is a significant difference in claim frequency between Women and Men (p < {alpha}).")
                    else:
                        print(
                            f"  Fail to reject H₀: No significant difference in claim frequency between Women and Men (p >= {alpha}).")
                else:
                    print(
                        "  Not enough variation in 'Gender' or 'HasClaim' for valid Chi-squared test.")
            else:
                print(
                    "  Not enough valid 'Gender' data (need at least 'Male' and 'Female') after cleaning for Chi-squared test.")
        else:
            print("  'Gender' column is missing or empty after cleaning.")

        # 4b. Claim Severity by Gender (Independent Samples t-test)
        if 'Gender' in df_claims_only.columns and not df_claims_only['Gender'].empty:
            gender_claims_only_data = df_claims_only[df_claims_only['Gender'].isin(
                ['Male', 'Female'])].copy()
            if not gender_claims_only_data.empty and len(gender_claims_only_data['Gender'].unique()) == 2:
                male_claims = gender_claims_only_data[gender_claims_only_data['Gender']
                                                      == 'Male']['TotalClaims']
                female_claims = gender_claims_only_data[gender_claims_only_data['Gender']
                                                        == 'Female']['TotalClaims']

                if len(male_claims) > 1 and len(female_claims) > 1:
                    # Perform independent t-test (assuming unequal variances by default, which is safer)
                    t_stat, p_val_severity_gender = stats.ttest_ind(
                        male_claims, female_claims, equal_var=False)
                    print(
                        f"Claim Severity by Gender (Independent t-test): p-value = {p_val_severity_gender:.4f}")
                    if p_val_severity_gender < alpha:
                        print(
                            f"  Reject H₀: There is a significant difference in claim severity between Women and Men (p < {alpha}).")
                    else:
                        print(
                            f"  Fail to reject H₀: No significant difference in claim severity between Women and Men (p >= {alpha}).")
                else:
                    print(
                        "  Not enough claim data for both 'Male' and 'Female' to perform Independent t-test.")
            else:
                print(
                    "  Not enough valid 'Gender' data (need both 'Male' and 'Female' with claims) after cleaning for t-test.")
        else:
            print("  'Gender' column is missing or empty in claims data after cleaning.")
        print("\n" + "="*80 + "\n")

        print("\n--- Interpretation Guidelines ---")
        print(
            f"For each test, compare the 'p-value' to the 'Significance level (alpha)' ({alpha}).")
        print(
            f"If p-value < {alpha}: Reject the Null Hypothesis (H₀). This means there's statistically significant evidence of a difference.")
        print(
            f"If p-value >= {alpha}: Fail to Reject the Null Hypothesis (H₀). This means there's no statistically significant evidence of a difference.")
        print("\nFor ANOVA tests, if the p-value is significant, a Tukey's HSD Post-hoc test is performed to identify which specific groups differ.")

    except FileNotFoundError:
        print(
            f"Error: The file was not found. Please ensure the CSV file is in the correct directory and the path is accurate.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
