import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_and_save(input_csv_path, output_csv_path, scaler_path):
    # Load data
    df = pd.read_csv(input_csv_path)
    
    # Menghapus kolom yang tidak dibutuhkan
    if 'loan_id' in df.columns:
        df = df.drop(columns='loan_id')

    # Menghapus whitespace pada nama kolom
    df.columns = df.columns.str.strip()

    # Menghapus whitespace pada data bertipe objek
    df_obj = df.select_dtypes(include='object')
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    # Normalisasi kolom numerik
    numeric_cols = df.select_dtypes(include='int64').columns.tolist()
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Mapping data kategorikal
    df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

    # Menyimpan hasil ke CSV
    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    preprocess_and_save(
        input_csv_path='./loan_approval_dataset.csv',
        output_csv_path='preprocessing/loan_preprocessing.csv',
    )
