//===================================================================================================================
//
// devapi_attr.cpp 	- C++ source code file for TANGO devapi class DeviceAttribute
//
// programmer(s) 	- Andy Gotz (goetz@esrf.fr)
//
// original 		- February 2002
//
// Copyright (C) :      2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015
//						European Synchrotron Radiation Facility
//                      BP 220, Grenoble 38043
//                      FRANCE
//
// This file is part of Tango.
//
// Tango is free software: you can redistribute it and/or modify it under the terms of the GNU
// Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Tango is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Tango.  If not, see <http://www.gnu.org/licenses/>
//
// $Revision$
//
//====================================================================================================================

// #if HAVE_CONFIG_H
// #include <ac_config.h>
// #endif

// #include <tango.h>

// using namespace CORBA;
#include "DeviceAttribute.h"

// namespace Tango
// {


void DeviceAttribute::deep_copy(const DeviceAttribute & source)
{
	w_dim_x = source.w_dim_x;
	w_dim_y = source.w_dim_y;

}



//-----------------------------------------------------------------------------
//
// DeviceAttribute::DeviceAttribute() - default constructor to create DeviceAttribute
//
//-----------------------------------------------------------------------------

DeviceAttribute::DeviceAttribute()
{
	w_dim_x = 0;
	w_dim_y = 0;
}


// }