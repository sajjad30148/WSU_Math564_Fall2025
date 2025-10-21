# ================================================================
# run_logger.py
# ---------------------------------------------------------------
# purpose
#     create and write iteration tables and summary files 
#
# outputs
#     iterations_<optimizer>_<ls>.txt   fixed width table
#     iterations_<optimizer>_<ls>.csv   csv with same columns
#     summary.txt                       run summary and x_final
#
# ================================================================
import os
from pathlib import Path
import csv
from datetime import datetime

class RunLogger:
    def __init__(self, out_dir, optimizer, line_search, alpha0 = 1.0,
                 max_iter = None, gtol = None, c1 = None, c2 = None, run_tag = None):
        
        # prepare folder and filenames
        self.out_dir = os.path.abspath(out_dir)
        # create output directory (parents=True to create any missing parents)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        base = f"iterations_{optimizer}_{line_search}"
        if run_tag:
            base = f"{base}_{run_tag}"

        self.txt_path = os.path.join(self.out_dir, f"{base}.txt")
        self.csv_path = os.path.join(self.out_dir, f"{base}.csv")
        self.sum_path = os.path.join(self.out_dir, f"{optimizer}_{line_search}_summary.txt")

        # open files
        self.txt_fh = open(self.txt_path, "w", newline = "")
        self.csv_fh = open(self.csv_path, "w", newline = "")
        self.csv_wr = csv.writer(self.csv_fh)

        # write headers
        # txt header matches the provided style
        self.txt_fh.write("date        time        iter           f         |g|        |ap|        |df|\n")
        # csv header
        self.csv_wr.writerow(["date", "time", "iter", "f", "gnorm", "ap_norm", "df_abs"])

        # immediate flush so files are never 0 KB
        self.txt_fh.flush()
        self.csv_fh.flush()

        # store settings for summary
        self.meta = {
            "optimizer": optimizer,
            "line_search": line_search,
            "max_iter": max_iter,
            "gtol": gtol,
            "c1": c1,
            "c2": c2,
            "alpha0": alpha0,
        }

        # running counters
        self.n_evals = 0  # objective evals inside line searches
        self.n_gevals = 0  # gradient evals if tracked by caller

    # formatting helpers to mirror the scientific notation seen in the example files
    @staticmethod
    def _fmt_e_signed(x, prec = 4):
        # used for f to get like +1.2417e+03
        return f"{x:+.{prec}e}"

    @staticmethod
    def _fmt_e(x, prec = 3):
        # used for norms and df to get like 5.659e+04
        return f"{x:.{prec}e}"

    @staticmethod
    def _today_strings():
        now = datetime.now()
        date_str = now.strftime("%d-%b-%Y")
        time_str = now.strftime("%H:%M:%S")
        return date_str, time_str

    def add_eval_counts(self, f_evals = 0, g_evals = 0):
        self.n_evals += int(f_evals)
        self.n_gevals += int(g_evals)

    def log_iter(self, k, f_val, gnorm, alpha, pnorm, df_abs):
        # compute ap norm like in the table
        ap_norm = abs(alpha) * pnorm

        date_str, time_str = self._today_strings()

        # write txt row
        # field widths chosen to visually match the example layout
        row = (
            f"{date_str:<12}"
            f"{time_str:<12}"
            f"{k:5d}"
            f"   {self._fmt_e_signed(f_val, prec = 4):>12}"
            f"   {self._fmt_e(gnorm, prec = 3):>9}"
            f"   {self._fmt_e(ap_norm, prec = 3):>9}"
            f"   {self._fmt_e(df_abs, prec = 3):>9}\n"
        )
        self.txt_fh.write(row)

        # write csv row with raw numbers
        self.csv_wr.writerow([date_str, time_str, int(k), float(f_val),
                              float(gnorm), float(ap_norm), float(df_abs)])

    def finalize(self, success, n_iter, f_final, g_final, x_final):
        # close iteration logs first to flush
        self.txt_fh.flush()
        self.csv_fh.flush()

        # write summary.txt in the same key order and style as the example
        with open(self.sum_path, "w", newline = "") as fh:
            fh.write(f"optimizer    : {self.meta.get('optimizer')}\n")
            fh.write(f"line search  : {self.meta.get('line_search')}\n")
            if self.meta.get("max_iter") is not None:
                fh.write(f"max_iter     : {self.meta.get('max_iter')}\n")
            if self.meta.get("gtol") is not None:
                fh.write(f"gtol         : {self.meta.get('gtol')}\n")
            if self.meta.get("c1") is not None:
                fh.write(f"c1           : {self.meta.get('c1')}\n")
            if self.meta.get("c2") is not None:
                fh.write(f"c2           : {self.meta.get('c2')}\n")
            if self.meta.get("alpha0") is not None:
                fh.write(f"alpha0       : {self.meta.get('alpha0')}\n")
            fh.write("\n")
            fh.write(f"success      : {bool(success)}\n")
            fh.write(f"n_iter       : {int(n_iter)}\n")
            # n_evals here records objective evals inside line searches
            fh.write(f"n_evals      : {int(self.n_evals)}\n")
            fh.write(f"f_final      : {self._fmt_e_signed(float(f_final), prec = 8)}\n")
            fh.write(f"|g|_final    : {self._fmt_e(float(g_final), prec = 3)}\n")
            fh.write("\n")
            fh.write("x_final:\n")
            # print vector in one line like the example, space separated
            fh.write("[ " + " ".join(f"{float(x): .6f}" for x in x_final) + " ]\n")

    def close(self):
        try:
            self.txt_fh.close()
        except Exception:
            pass
        try:
            self.csv_fh.close()
        except Exception:
            pass
