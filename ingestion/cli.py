import click
import yaml
from loader import BigQueryLoader


@click.command()
@click.option("--file", default=None, help="Path to CSV file")
@click.option("--schema", default=None, help="Schema name")
@click.option("--table", default=None, help="Target table name")
@click.option("--mode", default=None, help="append or truncate")

def load(file, schema, table, mode):

    with open("ingestion/config.yaml") as f:
        config = yaml.safe_load(f)

    # Prompt only if not supplied
    if not file:
        file = input("Enter CSV file path: ").strip()

    if not schema:
        schema = input("Enter schema name: ").strip()

    if not table:
        table = input("Enter target table name: ").strip()

    write_mode = (
        mode
        or input(
            f"Write mode [{config['write_mode']}]: "
        ).strip().lower()
        or config["write_mode"]
    )

    loader = BigQueryLoader(
        project_id=config["project_id"],
        dataset=config["dataset"],
        write_mode=write_mode
    )

    loader.load_csv(
        file_path=file,
        table_name=table,
        schema_name=schema
    )


if __name__ == "__main__":
    load()