 
# Create a virtual environment and install dependencies

## remove existing virtual environment
```shell
rm -rf .venv
```

## Create the new .venv using Python 3.13
```shell
python3.13 -m venv .venv
```

## Activate the virtual environment
```shell
source .venv/bin/activate
```

## Verify it's on Python 3.13
```shell
python3 --version
```

## Install dependencies
```shell
python -m pip install -r requirements.txt
```

## Upgrade pip
```shell
python -m pip install --upgrade pip
```