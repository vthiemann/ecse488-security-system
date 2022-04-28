import io
from flask import Flask, redirect, url_for, render_template, Response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", name="ECSE 488")

if __name__ == "__main__":
    app.run(debug=True)