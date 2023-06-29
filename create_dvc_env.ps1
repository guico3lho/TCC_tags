# Set-ExecutionPolicy RemoteSigned
Remove-Item -Recurse -Force -Path  "./experiment"

# copy pipeline to experiment when experiments are done
cp -r pipeline experiment

# convert .ipynb to .py for automate tests
jupyter nbconvert --to script experiment/*.ipynb;
jupyter nbconvert --to script experiment/supervised_deep_models/*.ipynb;
jupyter nbconvert --to script experiment/supervised_trad_models/*.ipynb

# remove .ipynb files
rm experiment/*.ipynb;
rm experiment/supervised_deep_models/*.ipynb;
rm experiment/supervised_trad_models/*.ipynb;
