#!/usr/bin/env bash

is_install=0
is_force=0

for opt in $@
do
    if [[ "$opt" == "-f" ]]; then
        is_force=1
    elif [[ "$opt" == "-i" ]]; then
        is_install=1
    else
        echo "not such option: $opt !"
        exit -1
    fi
done

project_name=$(xmllint --xpath '//package/name/text()' package.xml)

export PACKAGE_VERSION=$(xmllint --xpath '//package/version/text()' package.xml)
export PACKAGE_MAINTAINER=$(xmllint --xpath '//package/maintainer/text()' package.xml)
export PACKAGE_CONTACT=$(xmllint --xpath 'string(//package/maintainer/@email)' package.xml)
export PACKAGE_DESCRIPTION=$(xmllint --xpath '//package/description/text()' package.xml)


if [[ ${is_force} != 1 ]]; then
    if [[ -z "$(git status --porcelain)" ]]; then
        echo "git is clean!"
    else
        echo "git is not clean, please commit first!"
        exit -1
    fi
fi



if [[ -d build ]] ; then
    rm -rf build/
fi

mkdir build
cd build


CORE_COUNT=`grep processor /proc/cpuinfo | wc -l`
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/raccoon -DCATKIN_BUILD_BINARY_PACKAGE="1"
make -j$((CORE_COUNT + 2))

if ! [[ $? = 0 ]] ; then
    exit -1
fi


make package -j$((CORE_COUNT + 2))

if [[ ${is_install} == 1 ]]; then
    deb_file="$(ls | grep *.deb)"
    sudo dpkg -i "$deb_file"
fi

