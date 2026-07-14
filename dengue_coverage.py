"""Report per-country weekly coverage in the OpenDengue extract, before any country is locked."""
from to_schema import dengue_country_coverage

CSV = r"data/Final datasets/OpenDengue_Best_Spacial.csv"

cov = dengue_country_coverage(
    CSV,
    level="Admin1",        # "Admin1" | "Admin2"
    t_res_filter="Week",   # "Week" | "Month" | None for both
)
cov.to_csv("dengue_coverage_admin1.csv", index=False)

print(cov.to_string(index=False))
