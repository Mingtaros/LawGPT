# Indonesian Law GPT
Simple AI-based application to help Indonesians know more about their ongoing case, what constitutional law is affected, etc.

## Data Sources
<table>
    <tr>
        <th>Name</th><th>Data Format</th><th>Source</th>
    </tr><tr>
        <td>Indonesian Constitution</td><td>PDF</td><td>https://www.mkri.id/public/content/infoumum/regulation/pdf/UUD45%20ASLI.pdf</td>
    </tr>
</table>

## How to Run Crawler
1. Delete the file `data/peraturan_go_id_output.jsonl`.
2. Go back to root directory and then go to `src/crawler/`.
3. Run `scrapy crawl peraturangoid -o ../../data/peraturan_go_id_output.jsonl`.
4. Wait until it finishes.
5. After it finishes, go to `data/peraturan_go_id_output.jsonl` to see the outputs.

## How to Run Main Module
1. Install necessary packages `pip install -r requirements.txt`.
2. Make `.env` file with the example, and fill in the necessary keys.
3. Turn on Docker service and Run Posgres VectorDB:
    ```shell
    docker run -d \
            -e POSTGRES_DB=ai \
            -e POSTGRES_USER=ai \
            -e POSTGRES_PASSWORD=ai \
            -e PGDATA=/var/lib/postgresql/data/pgdata \
            -v pgvolume:/var/lib/postgresql/data \
            -p 5532:5432 \
            --name pgvector \
            agnohq/pgvector:16
    ```
4. From root, run `streamlit run src/app.py`. If successful, this will open your browser, `localhost:8501`
5. Wait until everything loads. When it finishes, a text input box appears.
6. Input with your case and see the response.
