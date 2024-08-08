from openpyxl import Workbook
from typing import List, Any


class ExcelExporter:
    def __init__(self, file_path: str, save_interval: int = 1000):
        self.file_path = file_path
        self.workbook = Workbook()
        self.worksheet = self.workbook.active
        self.save_interval = save_interval
        self.rows_written = 0

    def write_header(self, columns: List[str]):
        self.worksheet.append(columns)
        self.rows_written += 1

    def write_row(self, row: List[Any]):
        # Append a row to the Excel file and save periodically
        self.worksheet.append(row)
        self.rows_written += 1

        if self.rows_written % self.save_interval == 0:
            self.save()
            print(f"Auto-saved after {self.rows_written} rows.")

    def save(self):
        # Save the Excel file
        self.workbook.save(self.file_path)
        print(f"File saved to {self.file_path}")
