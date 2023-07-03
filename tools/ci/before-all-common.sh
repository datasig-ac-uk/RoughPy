#
# Copyright (c) 2023 Datasig Developers. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

echo $OSTYPE
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  yum install -y curl zip unzip tar gmp-devel

  # manylinux doesn't have mono, we'll have to install it
  rpmkeys --import "http://keyserver.ubuntu.com/pks/lookup?op=get&search=0x3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF"
  su -c 'curl https://download.mono-project.com/repo/centos7-stable.repo | tee /etc/yum.repos.d/mono-centos7-stable.repo'
  yum install -y mono-complete

  MONO_EXE=mono
elif [[ "$OSTYPE" == "darwin"* ]]; then
  brew install libomp
  MONO_EXE=mono
elif [[ "$OSTYPE" == "msys"* ]]; then
  MONO_EXE=""
else
  exit 1
fi

git clone https://github.com/Microsoft/vcpkg.git
bash vcpkg/bootstrap-vcpkg.sh -disableMetrics

vcpkg_root=./vcpkg

#if [ -n "$GITHUB_TOK" ]; then
#  # If the token is defined, set up binary caching
#  $MONO_EXE `$vcpkg_root/vcpkg fetch nuget | tail -n 1` \
#    sources add \
#    -source "https://nuget.pkg.github.com/datasig-ac-uk/index.json" \
#    -storepasswordincleartext \
#    -name "GitHub" \
#    -username "datasig-ac-uk" \
#    -password "$GITHUB_TOK"
#  $MONO_EXE `$vcpkg_root/vcpkg fetch nuget | tail -n 1` \
#    setapikey "$GITHUB_TOK" \
#    -source "https://nuget.pkg.github.com/datasig-ac-uk/index.json"
#fi
