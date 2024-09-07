import setuptools

setuptools.setup(
    name="syn_tracker",
    version="0.0.1",
    description="Tools to detect and track flies",
    url="",
    author="Weheliye Hashi",
    author_email="w.weheliye@ic.ac.uk",
    license="MIT",
    packages=setuptools.find_packages(),
    zip_safe=False,
    entry_points={"console_scripts": ["process_data=" "syn_tracker.main_process:main"]},
)
