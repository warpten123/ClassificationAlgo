# How to Set-Up Dependencies

1. Create a virtual environment using Pipenv:

<pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>sh</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-sh">python -m venv myenv
</code></div></div></pre>

Note: Replace myenv name

This will create a virtual environment using Pipenv with the specified Python version.

2. Install the packages using Pipenv:

Certainly! You can install the packages listed in the requirements.txt file using pipenv with the following command:

<pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs">pipenv install -r requirements.txt</code></div></div></pre>

3. Install `pipenv` using `pip`:

<pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>sh</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-sh">pip install pipenv
</code></div></div></pre>

4. Navigate to the root directory of your project in the terminal.
5. Initialize a new `pipenv` environment and install the project's dependencies:

<pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>sh</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-sh">pipenv install
</code></div></div></pre>

This will automatically scan your project's code for dependencies listed in import statements, and generate a `Pipfile` and a `Pipfile.lock` file in the project's root directory, which specify the dependencies and their exact versions used in your project.

6. After installation, you can generate the Pipenv lock file that lists all the installed packages and their versions using the following command:

<pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>sh</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-sh">pipenv lock</code></div></div></pre>

7. Run your project's API or other scripts within the `pipenv` environment:

<pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>sh</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-sh">pipenv run python Python_Backend/main.py
</code></div></div></pre>

This ensures that the dependencies installed in the `pipenv` environment are used when running your project's code.

8. To share your project with others, you can provide the `Pipfile` and `Pipfile.lock` files, which contain the list of dependencies and their versions. Others can then use `pipenv` to install the same dependencies with the exact versions specified in the `Pipfile.lock` file in their own virtual environment.

Alternatively, you can also use `poetry`, another popular Python package management tool, which follows a similar approach but uses a different format (`pyproject.toml` and `poetry.lock`) for specifying dependencies and versions.

Using `pipenv` or `poetry` can help you automatically generate a list of dependencies to be installed based on your project's code, ensuring consistent and reproducible builds across different environments.

# Setup for OCR

### ImageMagick

The error message suggests that the ImageMagick library, which is required by the pdfplumber library to handle image extraction from PDFs, is not installed on your system.

You can follow the instructions provided in the link mentioned in the error message (https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-windows) to install ImageMagick on Windows.

Here are the general steps to install ImageMagick on Windows:

Download the ImageMagick installer for Windows from the official ImageMagick website (https://imagemagick.org/script/download.php#windows).
Run the installer and follow the installation instructions.
Make sure to select the "Install legacy utilities (e.g. convert)" option during the installation process.
Once the installation is complete, add the ImageMagick installation directory to the system's PATH environment variable. This can usually be done through the advanced system settings or environment variables settings in Windows.
Restart any open command prompt or terminal windows to apply the changes.
After installing ImageMagick, you should be able to use the pdfplumber library to extract text and images from PDFs without encountering the "MagickWand shared library not found" error.

### Ghost

Ghostscript is a dependency of ImageMagick and is required for handling PDFs. In order to fix this error, you need to make sure that Ghostscript is installed on your system and its executable is accessible.

You can download and install Ghostscript from the official Ghostscript website (https://www.ghostscript.com/download/gsdnld.html) or through a package manager if you are using a package-based system like Ubuntu.

After installing Ghostscript, you may need to add the directory containing the gswin64c.exe executable to your system's PATH environment variable, similar to how you added the ImageMagick installation directory to the PATH as mentioned in the previous response. This will allow ImageMagick to locate and use the Ghostscript executable when needed.

Once Ghostscript is installed and configured correctly, you should be able to run your code without encountering the "FailedToExecuteCommand" error.

# Links

install tesseract
-> https://github.com/UB-Mannheim/tesseract/wiki
install imagemagick
-> https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-windows
-> https://imagemagick.org/script/download.php#windows
