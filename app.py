"""Interface layer: Streamlit chat UI. Calls only the agent layer, never the
model or tools directly, and catches every exception the layers below raise."""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
