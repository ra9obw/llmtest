
//+==================================================================================================================
//
// DeviceAttribute.h - include file for TANGO device api class DeviceAttribute
//
//
// Copyright (C) :      2012,2013,2014,2015
//						European Synchrotron Radiation Facility
//                      BP 220, Grenoble 38043
//                      FRANCE
//
// This file is part of Tango.
//
// Tango is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Tango is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with Tango.
// If not, see <http://www.gnu.org/licenses/>.
//
// $Revision: 20437 $
//
//+==================================================================================================================

#ifndef _DEVICEATTRIBUTE_H
#define _DEVICEATTRIBUTE_H


/****************************************************************************************
 * 																						*
 * 					The DeviceAttribute class											*
 * 					-------------------------											*
 * 																						*
 ***************************************************************************************/


/**
 * Fundamental type for sending an dreceiving data to and from device attributes.
 *
 * This is the fundamental type for sending and receiving data to and from device attributes. The values can be
 * inserted and extracted using the operators << and >> respectively and insert() for mixed data types. There
 * are two ways to check if the extraction operator succeed :
 * <ul>
 * <li> 1. By testing the extractor operators return value. All the extractors operator returns a boolean value set
 * to false in case of problem.
 * <li> 2. By asking the DeviceAttribute object to throw exception in case of problem. By default, DeviceAttribute
 * throws exception :
 *    <ol>
 *    <li> When the user try to extract data and the server reported an error when the attribute was read.
 *    <li> When the user try to extract data from an empty DeviceAttribute
 *    </ol>
 * </ul>
 *
 * <B>For insertion into DeviceAttribute instance from TANGO CORBA sequence pointers, the DeviceAttribute
 * object takes ownership of the pointed to memory. This means that the pointed
 * to memory will be freed when the DeviceAttribute object is destroyed or when another data is
 * inserted into it. The insertion into DeviceAttribute instance from TANGO CORBA sequence reference copy
 * the data into the DeviceAttribute object.\n
 * For extraction into TANGO CORBA sequence types, the extraction method consumes the
 * memory allocated to store the data and it is the caller responsibility to delete this memory.</B>
 *
 * $Author: taurel $
 * $Revision: 1 $
 *
 * @headerfile tango.h
 * @ingroup Client
 */

class DeviceAttribute
{

public :

	void deep_copy(const DeviceAttribute &);
	DeviceAttribute();

	int 				w_dim_x;
	int 				w_dim_y;

};

#endif /* _DEVICEATTRIBUTE_H */
