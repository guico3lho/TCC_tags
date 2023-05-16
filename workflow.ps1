# Set-ExecutionPolicy RemoteSigned
# Remove-Item -Recurse -Force -Path  "./experiment"
# cp -r pipeline experiment
jupyter nbconvert --to script experiment/*.ipynb;
jupyter nbconvert --to script experiment/supervised_deep_models/*.ipynb;
jupyter nbconvert --to script experiment/supervised_traditional_models/*.ipynb
rm experiment/*.ipynb;
rm experiment/supervised_deep_models/*.ipynb;
rm experiment/supervised_traditional_models/*.ipynb;
